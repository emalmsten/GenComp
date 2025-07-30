from torch.utils.data import Dataset
import numpy as np
from lego.lego_util import get_mpd_path, get_translations, get_rotations, normalize


class LegoDataset(Dataset):
    def __init__(self, cfg, bricks_per_sample, start=0, end=-1, manual_file_paths=None):
        self.end = end
        self.start = start
        if end == -1:
            end = cfg.data.dataset.num_samples

        # When talking about shapes N = num_samples, B = num_bricks_in_sample, C = num_corners
        print(f"Making dataset: {bricks_per_sample}_{start}_{end}")
        l_cfg = cfg.ltron_params
        self.cfg = cfg

        self.num_corners = l_cfg.bbox_corners
        self.max_num_points = l_cfg.max_num_points
        self.bricks_per_sample = bricks_per_sample
        self.normalization_factor = l_cfg.normalization_factor
        self.ids = [i for i in range(start, end)]

        # If wanting to use own mpd files instead of ltron
        if manual_file_paths is None:
            file_paths = [get_mpd_path(idx, self.bricks_per_sample, self.cfg) for idx in self.ids]
        else:
            file_paths = manual_file_paths

        # Get the data and conditionals
        self.datas, self.conds = self.create_dataset(file_paths)


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = np.transpose(self.datas[idx])
        cond = self.conds[idx]

        temp_cond = {
            'self_mask': cond['self_mask'],
            'gen_mask': cond['gen_mask'],
            'poly': cond['puzzle']['poly'],
            'src_key_padding_mask': np.concatenate([np.zeros(34), np.ones(100 - 34)]),
            'id': self.ids[idx],
            'num_bricks': self.bricks_per_sample,
        }

        return data, temp_cond


    def create_dataset(self, file_paths):
        """Creates the dataset from the file paths"""
        # Breaks the code if imported at the top, since it needs paths for some reason
        from ltron.bricks.brick_scene import BrickScene

        l_cfg = self.cfg.ltron_params
        scene = BrickScene(
            renderable=False,
            collision_checker=False,
            track_snaps=True)

        print("Getting all bricks from each construction")
        samples = [self.get_all_bricks(file_path, scene) for file_path in file_paths]
        scene.instances.clear()

        print("Getting the information from the bricks")
        # shape (N, B, 4, 4), (N, B, C, 3)
        transforms, bboxes = get_transforms_and_bboxes(samples, self.num_corners)

        print("Finding the rotations and translations")
        rotations = get_rotations(transforms)  # shape (N, B, 3)
        translations = get_translations(transforms)

        if l_cfg.find_normalization_factor:
            print("Finding normalization factor")
            # Simply writes it to file to be gotten at another point,
            # todo this could be fixed if going through all datasets at once but currently not supported
            self.find_normalization_factor(translations)

        norm_translations = normalize(translations, self.normalization_factor)
        norm_bboxes = normalize(bboxes, self.normalization_factor)
        norm_rotations = normalize(rotations, np.pi)

        datas = self.repeat_and_flatten(np.concatenate([norm_translations, norm_rotations], axis=-1)) # Shape: (N, max_num_points, 4)
        conds = self.get_conditionals(norm_bboxes) # List of dictionaries of size N

        return datas, conds

    def get_all_bricks(self, file_path, scene):
        """Gets all bricks from a mpd file"""
        scene.instances.clear()
        scene.import_ldraw(file_path)
        return list(scene.instances.instances.values())


    def repeat_and_flatten(self, datas):
        # Add a new axis for repetition
        combined = datas[:, :, np.newaxis, :]  # Shape: (N, B, 1, 4)

        # Repeat the combined array 'C' times along the new axis
        combined_repeated = np.repeat(combined, self.num_corners, axis=2)  # Shape: (N, B, C, 4)

        # Flatten along the second and third axes
        datas = combined_repeated.reshape(combined_repeated.shape[0], -1, combined_repeated.shape[3]) # Shape: (N, B*C, 4)

        # pad B*C to max_num_points
        datas = np.concatenate([datas, np.zeros((datas.shape[0], self.max_num_points - datas.shape[1],
                                                 datas.shape[2]))], axis=1) # Shape: (N, max_num_points, 4)

        return datas

    def transform_bboxes(self, corners, transformations):
        """Not used but I'll leave it just in case, to transform the corners of the bboxes"""

        # Convert corners to homogeneous coordinates
        ones = np.ones(corners.shape[:-1] + (1,))
        homogeneous_corners = np.concatenate([corners, ones], axis=-1)

        # Prepare for batch matrix multiplication
        transformations_expanded = transformations[..., np.newaxis, :, :]
        homogeneous_corners_expanded = homogeneous_corners[..., np.newaxis]

        # Perform batch matrix multiplication
        transformed_corners = np.matmul(transformations_expanded, homogeneous_corners_expanded)

        # Reshape and extract transformed coordinates
        transformed_corners = transformed_corners[..., 0]
        transformed_corners = transformed_corners[..., :3]

        return transformed_corners


    def get_conditionals(self, bboxes):
        """Gets the conditionals for the dataset, just the corners of the bboxes for now"""
        # todo could use some torch optimization and clean up.
        #  Taken from puzzlefusion code and cleaned up A LOT, but could get rid of some looping

        print("Getting conditionals")
        # Will contain one dictionary for each construction
        construction_dicts = []
        num_samples = len(bboxes)
        num_cor = self.num_corners
        total_bricks = num_samples * self.bricks_per_sample

        # Compute corner_bounds for all bricks, used for attention
        corner_bounds = [[i * num_cor, (i + 1) * num_cor] for i in range(total_bricks)]
        cb_len = 0

        # Loop through all samples/constructions
        for sample_idx, construction in enumerate(bboxes):
            if sample_idx % 5000 == 0:
                print(f"done with {sample_idx} out of {num_samples}")

            # Here snaps could potentially also be added
            brick_dicts = {'poly': bboxes[sample_idx]}

            # Create the self mask, makes sure the bricks don't pay attention to themselves given they are repeated
            cb_len += self.bricks_per_sample
            self_mask = np.ones((self.max_num_points, self.max_num_points))
            for i in range(cb_len):
                self_mask[corner_bounds[i][0]:corner_bounds[i][1], corner_bounds[i][0]:corner_bounds[i][1]] = 0

            # Creates the gen mask, makes sure no attention is paid to padding
            flat_ar_length = sum(bbox.shape[0] for bbox in bboxes[sample_idx])
            gen_mask = np.ones((self.max_num_points, self.max_num_points))
            gen_mask[:flat_ar_length, :flat_ar_length] = 0

            # Pads the conditionals to max_num_points
            for key in brick_dicts:
                brick_dicts[key] = np.concatenate(brick_dicts[key], axis=0)
                current_length = brick_dicts[key].shape[0]
                padding_length = self.max_num_points - current_length

                # Determine the padding shape based on the array's dimensions
                pad_width = [(0, padding_length)] + [(0, 0)] * (brick_dicts[key].ndim - 1)

                # Pad the array with zeros
                brick_dicts[key] = np.pad(brick_dicts[key], pad_width, mode='constant', constant_values=0)

            # Each sample/construction has a dictionary of this format
            sample_dict = {'puzzle': brick_dicts, 'self_mask': self_mask, 'gen_mask': gen_mask}
            construction_dicts.append(sample_dict)

        return construction_dicts

    def find_normalization_factor(self, translations):
        """Finds the normalization factor for the dataset"""
        max_value = np.abs(translations).max()
        file = "./code/lego/normalization_factor.txt"
        # append to file, make if it doesn't exist
        with open(file, "a+") as f:
            f.write(f"bricks: {self.bricks_per_sample} start: {self.start} end: {self.end} normalization_factor: {max_value}\n")
        return


def get_transforms_and_bboxes(samples, num_corners):
    """Gets the transforms and bboxes from the samples"""
    assert num_corners == 2 or num_corners == 8, "Only 2 or 8 corners supported"

    transforms = np.array([[brick.transform for brick in brick_set] for brick_set in samples])

    if num_corners == 2:
        bboxes = np.array([[brick.brick_shape.bbox for brick in brick_set] for brick_set in samples])
    else:
        bboxes = np.array([[brick.brick_shape.bbox_vertices for brick in brick_set] for brick_set in samples])
        bboxes = np.transpose(bboxes, (0, 1, 3, 2))
        # Remove last column since it is always 1
        bboxes = bboxes[:, :, :, :-1]

    return transforms, bboxes






