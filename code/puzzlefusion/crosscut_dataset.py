import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
import os
import io
# import cv2 as cv
import csv
from tqdm import tqdm
# from shapely import geometry as gm
from collections import defaultdict
from glob import glob
from scipy import ndimage
# from puzzlefusion.embedder.model import get_model
# from embedder.model import get_model
import torchvision


import PIL.Image as Image
import drawsvg as drawsvg
import cairosvg
import webcolors



def rotate_points(points, indices):
    indices = np.argmax(indices,1)
    indices[indices==0] = 1000
    unique_indices = np.unique(indices)
    num_unique_indices = len(unique_indices)
    rotated_points = np.zeros_like(points)
    rotation_angles = []
    for i in unique_indices:
        idx = (indices == i)
        selected_points = points[idx]
        rotation_degree = 0 if i==1 else (np.random.rand() * 360)
        # rotation_angle = 0 
        # rotation_angle = 0 if i==0 else (np.random.randint(4) * 90)
        rotation_angle = np.deg2rad(rotation_degree)
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)], # this is selected for return
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        rotated_selected_points = np.matmul(rotation_matrix, selected_points.T).T
        rotated_points[idx] = rotated_selected_points
        # rotation_matrix[0,1] = 1 if rotation_angle<np.pi else -1
        rotation_angles.extend(rotation_matrix[0:1].repeat(rotated_selected_points.shape[0], axis=0))
    return rotated_points, rotation_angles, rotation_degree


def load_crosscut_data(
    batch_size,
    set_name,
    rotation,
    use_image_features,
):
    """
    For a dataset, create a generator over (shapes, kwargs) pairs.
    """
    print(f"loading {set_name} of crosscut...")
    deterministic = False if set_name=='train' else True
    dataset = CrosscutDataset(set_name, rotation=rotation, use_image_features=use_image_features)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
        )
    while True:
        yield from loader

# training: 3 'lines' dir each with 35 000 puzzles
class CrosscutDataset(Dataset):
    def __init__(self, set_name, rotation, use_image_features, data_path='./datasets/puzzlefusion', numSamples=100, **kwargs):
        super().__init__()
        duplicate = False
        self.use_image_features = use_image_features
        get_one_hot = lambda x, z: np.eye(z)[x]
        max_num_points = 100
        base_dir = f'{data_path}/cross_cut/{set_name}_poly_data'
        img_base_dir = f'{data_path}/poly_data'#doesnt existm skip to processed in line 91
        self.set_name = set_name
        self.rotation = rotation
        self.sample_files = []
        sampled = 0

        # if self.use_image_features:
        #     device = "cuda" if th.cuda.is_available() else "cpu"
        #     model = get_model('./puzzle_fusion/embedder/ckpts/new_exp_128_losscolor/model.pt', use_gpu=True)
        #     model.eval()
        #     transform = torchvision.transforms.ToTensor()
        #import pdb ; pdb.set_trace()
        lines_dir = glob(f'{base_dir}/*')
        # sample amount of samples from each line directory (3 lines) with remeinder taken from line dir 1.
        for line_idx, directory in enumerate(lines_dir[:3 if numSamples > 3 else numSamples]):
            puzzles = glob(f'{directory}/*')
            puzzles = [puzzle_name.replace('\\', '/') for puzzle_name in puzzles]
            num_samples_from_line =  1 if numSamples < 3 else (numSamples//3 if line_idx < 2 else numSamples//3 + numSamples%3)
            for puzzle_name in tqdm(puzzles[:num_samples_from_line]):
                sampled += 1

                if self.use_image_features:
                    image_puzzle_name = f"{img_base_dir}/_puzzle_name_{puzzle_name.split('/')[4]}_{puzzle_name.split('/')[5]}.npz"
                    if not os.path.isfile(image_puzzle_name):
                        continue
                    if os.path.isfile(f"{data_path}/cross_cut/processed/{puzzle_name.split('/')[4]}_{puzzle_name.split('/')[5]}.npz"):
                        self.sample_files.append(f"{data_path}/cross_cut/processed/{puzzle_name.split('/')[4]}_{puzzle_name.split('/')[5]}.npz")
                        continue
                if self.use_image_features:
                    try:
                        img = np.load(image_puzzle_name)
                    except Exception:
                        continue
                with open(f'{puzzle_name}/ground_truth_puzzle.csv') as csvfile:
                ##### For having different noises change the above line to one of these below ###
                # with open(f'{puzzle_name}/err3_n.csv') as csvfile: #### error3
                # with open(f'{puzzle_name}/err2_n.csv') as csvfile: #### error2
                # with open(f'{puzzle_name}/err1_n.csv') as csvfile: #### error2
                    reader = csv.reader(csvfile, delimiter=',')
                    puzzle_dict = defaultdict(list)
                    puzzle = []
                    for row in reader:
                        if row[0] == 'piece':
                            continue
                        puzzle_dict[float(row[0])].append([float(row[1]),float(row[2])])
                    for piece in puzzle_dict.values():
                        piece = np.array(piece) / 100. - 0.5 # [[x0,y0],[x1,y1],...,[x15,y15]] and map to 0-1 - > -0.5, 0.5
                        piece = piece * 2 # map to [-1, 1]
                        center = np.mean(piece, 0)
                        piece = piece - center
                        if self.use_image_features:
                            puzzle.append({'poly': piece, 'center': center, 'img': img[str(len(puzzle))]})
                        else:
                            puzzle.append({'poly': piece, 'center': center})
                    if duplicate:
                        num_duplicates = np.random.randint(3)+1
                        for d_indx in range(num_duplicates):
                            duplicate_idx = np.random.randint(len(puzzle))
                            puzzle.append(puzzle[duplicate_idx])
                start_points = [0]
                for i in range(len(puzzle)-1):
                    start_points.append(start_points[-1]+len(puzzle[i]['poly']))
                with open(f'{puzzle_name}/ground_truth_rels.csv') as csvfile:
                ##### For having different noises change the above line to one of these below ###
                # with open(f'{puzzle_name}/err3_n.csv') as csvfile: #### error3
                # with open(f'{puzzle_name}/err2_n.csv') as csvfile: #### error2
                # with open(f'{puzzle_name}/err1_n.csv') as csvfile: #### error2
                    reader = csv.reader(csvfile, delimiter=',')
                    rels = []
                    for row in reader:
                        if row[0] == 'piece1':
                            continue
                        [p1, e1, p2, e2] = [int(x) for x in row]
                        p11 = puzzle[p1]['poly'][e1]+puzzle[p1]['center']
                        p12 = puzzle[p1]['poly'][(e1+1)%len(puzzle[p1]['poly'])] + puzzle[p1]['center']
                        p21 = puzzle[p2]['poly'][e2]+puzzle[p2]['center']
                        p22 = puzzle[p2]['poly'][(e2+1)%len(puzzle[p2]['poly'])] + puzzle[p2]['center']
                        if np.abs(p11-p21).sum()<np.abs(p11-p22).sum():
                            rels.append([start_points[p1]+e1, start_points[p2]+e2])
                            rels.append([start_points[p1]+(e1+1)%(len(puzzle[p1]['poly'])), start_points[p2]+(e2+1)%(len(puzzle[p2]['poly']))])
                        else:
                            rels.append([start_points[p1]+e1, start_points[p2]+(e2+1)%(len(puzzle[p2]['poly']))])
                            rels.append([start_points[p1]+(e1+1)%(len(puzzle[p1]['poly'])), start_points[p2]+e2])
                    padding = np.zeros((100-len(rels), 2))
                    rels = np.concatenate((np.array(rels), padding), 0)

                p = puzzle
                puzzle_img = []
                puzzle = []
                corner_bounds = []
                num_points = 0
                for i, piece in enumerate(p):
                    poly = piece['poly']# array of corner points of piece [num corners, 2]
                    center = np.ones_like(poly) * piece['center']
                    if self.use_image_features:
                        img = piece['img']

                    # Adding conditions
                    num_piece_corners = len(poly)
                    piece_index = np.repeat(np.array([get_one_hot(len(puzzle)+1, 32)]), num_piece_corners, 0)
                    corner_index = np.array([get_one_hot(x, 32) for x in range(num_piece_corners)])

                    # Adding rotation
                    if self.rotation:
                        poly, angles, degree = rotate_points(poly, piece_index)
                        # print("poly: ", poly.shape)
                        # print("angles: ", len(angles))
                        # print("angles: ", angles)
                        # print("degree: ", degree)
                        if self.use_image_features:
                            img = ndimage.rotate(img, degree, reshape=False)
                           

                    # Adding images
                    if self.use_image_features:
                        puzzle_img.append(img)
                        inputs = transform(img).to(device).float()
                        image_features = model(inputs.unsqueeze(0), pred_image=False).reshape(1,-1)
                        image_features = image_features.expand(poly.shape[0], image_features.shape[1]).cpu().data.numpy()

                    # Src_key_padding_mask
                    padding_mask = np.repeat(1, num_piece_corners)
                    padding_mask = np.expand_dims(padding_mask, 1)


                    # Generating corner bounds for attention masks
                    connections = np.array([[i,(i+1)%num_piece_corners] for i in range(num_piece_corners)])
                    connections += num_points
                    corner_bounds.append([num_points, num_points+num_piece_corners])
                    num_points += num_piece_corners
                    if self.use_image_features:
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, image_features), 1)
                    else:
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections), 1)
                    puzzle.append(piece)
                
                puzzle_layouts = np.concatenate(puzzle, 0)
                if len(puzzle_layouts)>max_num_points:
                    assert False
                if self.use_image_features:
                    padding = np.zeros((max_num_points-len(puzzle_layouts), 73+128))
                else:
                    padding = np.zeros((max_num_points-len(puzzle_layouts), 73))
                gen_mask = np.ones((max_num_points, max_num_points))
                gen_mask[:len(puzzle_layouts), :len(puzzle_layouts)] = 0
                puzzle_layouts = np.concatenate((puzzle_layouts, padding), 0)
                self_mask = np.ones((max_num_points, max_num_points))

                for i in range(len(corner_bounds)):
                    self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[i][0]:corner_bounds[i][1]] = 0
                sample_dict = {'puzzle': puzzle_layouts, 'self_mask': self_mask, 'gen_mask': gen_mask, 'rels': rels, 'images': puzzle_img}

                np.savez_compressed(f"{data_path}/cross_cut/processed/{puzzle_name.split('/')[4]}_{puzzle_name.split('/')[5]}", **sample_dict)
                self.sample_files.append(f"{data_path}/cross_cut/processed/{puzzle_name.split('/')[4]}_{puzzle_name.split('/')[5]}.npz")
        self.num_coords = 4
        
        print("Numer of samples sampled: ", sampled)

        #self.sample_files = self.sample_files[:10000]
        self.samples = []
        for file in tqdm(self.sample_files, desc="loading processed dataset..."):
            sample = dict(np.load(file))
            sample.pop('images', None)
            self.samples.append(sample)
          
        
    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        # sample = np.load(self.sample_files[idx])
        sample = self.samples[idx]
        puzzle = sample['puzzle']
        # puzzle = self.samples[idx]
        arr = puzzle[:, :self.num_coords]
        polys = puzzle[:, self.num_coords:self.num_coords+2]
      
        cond = {
                'self_mask': sample['self_mask'],
                'gen_mask': sample['gen_mask'],
                # 'self_mask': self.self_masks[idx],
                # 'gen_mask': self.gen_masks[idx],
                'poly': polys,
                'corner_indices': puzzle[:, self.num_coords+2:self.num_coords+34],
                'room_indices': puzzle[:, self.num_coords+34:self.num_coords+66],
                'src_key_padding_mask': 1-puzzle[:, self.num_coords+66],
                'connections': puzzle[:, self.num_coords+67:self.num_coords+69],
                'rels': sample['rels'],
                # 'rels': self.rels[idx],
                }
        if self.use_image_features:
            cond['image_features'] = puzzle[:, -128:]
        arr = np.transpose(arr, [1, 0])
        return arr.astype(float), cond

if __name__ == '__main__':
    dataset = CrosscutDataset('train', rotation=True, use_image_features=False)

# Display puzzle_cut data to svg/png
ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
                        6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 9: '#1F849B', 10: '#727171',
                        11: '#785A67', 12:'#D3A2C7', 13: '#ff55a3',14 : '#d7e8fc', 15: '#ff91af' ,
                        16 :'#d71868', 17: '#d19fe8', 18: '#00cc99', 19: '#eec8c8', 20:'#739373'}

def unrotate_points(points, cos_theta, sin_theta):
    shape = points.shape
    theta = -th.atan2(-sin_theta, cos_theta)
    cos_theta = th.cos(theta)
    sin_theta = -th.sin(theta)
    # theta = th.acos(cos_theta)
    # sin_theta[sin_theta>0] = 1
    # sin_theta[sin_theta<0] = -1
    # theta = theta * sin_theta
    # theta = -theta
    sin_theta = th.sin(theta)
    cos_theta = th.cos(theta)

    rotation_matrix = th.stack([
        th.stack([cos_theta, -sin_theta]),
        th.stack([sin_theta, cos_theta])
    ])
    rotation_matrix = rotation_matrix.permute([2,3,4,0,1])
    points = points.reshape(-1, 2, 1)
    rotation_matrix = rotation_matrix.reshape(-1, 2, 2)
    rotated_points = th.bmm(rotation_matrix.double(), points.double())
    return rotated_points.reshape(shape)

def base_save_samples(sample, ext, model_kwargs, rotation, tmp_count, save_gif=False, save_edges=False, ID_COLOR=ID_COLOR, save_svg=False, output_path='code/puzzlefusion/outputs', nosave=False):
    if not save_gif:
        sample = sample[-1:]
    for k in range(sample.shape[0]):
        if rotation:
            rot_s_total=[]
            rot_c_total=[]
            for nb in range(model_kwargs[f'room_indices'].shape[0]):
                array_a = np.array(model_kwargs[f'room_indices'][nb])
                room_types = np.where(array_a == array_a.max())[1]
                room_types = np.append(room_types, -10)
                rot_s =[]
                rot_c =[]
                rt =0
                no=0
                for ri in range(len(room_types)):
                    if rt!=room_types[ri]:
                        for nn in range(no):
                            rot_s.append(np.array(rot_s_tmp).mean())
                            rot_c.append(np.array(rot_c_tmp).mean())
                        rt=room_types[ri]
                        no=1
                        rot_s_tmp = [sample[k:k+1,:,:,3][0][nb][ri].data.numpy()]
                        rot_c_tmp = [sample[k:k+1,:,:,2][0][nb][ri].data.numpy()]
                    else:
                        no+=1
                        rot_s_tmp.append(sample[k:k+1,:,:,3][0][nb][ri].data.numpy())
                        rot_c_tmp.append(sample[k:k+1,:,:,2][0][nb][ri].data.numpy())
                while len(rot_s)<100:
                    rot_s.append(0)
                    rot_c.append(0)
                rot_s_total.append(rot_s)
                rot_c_total.append(rot_c)
            poly = unrotate_points(model_kwargs['poly'].unsqueeze(0),th.unsqueeze(th.Tensor(rot_c_total),0), th.unsqueeze(th.Tensor(rot_s_total),0))
            # poly = rotate_points(model_kwargs['poly'].unsqueeze(0), sample[k:k+1,:,:,2], sample[k:k+1,:,:,3])
        else:
            poly = model_kwargs['poly'].unsqueeze(0)


        center_total = []
        for nb in range(model_kwargs[f'room_indices'].shape[0]):
            array_a = np.array(model_kwargs[f'room_indices'][nb])
            room_types = np.where(array_a == array_a.max())[1]
            room_types = np.append(room_types, -10)
            center =[]
            rt =0
            no=0
            for ri in range(len(room_types)):
                if rt!=room_types[ri]:
                    for nn in range(no):
                        center.append(np.array(center_tmp).mean(0))
                    rt=room_types[ri]
                    no=1
                    center_tmp = [sample[k:k+1,:,:,:2][0][nb][ri].data.numpy()]
                else:
                    no+=1
                    center_tmp.append(sample[k:k+1,:,:,:2][0][nb][ri].data.numpy())
            while len(center)<100:
                center.append([0, 0])
            center_total.append(center)

        sample[k:k+1,:,:,:2] = th.Tensor(center_total) + poly
        # sample[k:k+1,:,:,:2] = sample[k:k+1,:,:,:2] + poly
    sample = sample[:,:,:,:2]
    for i in range(sample.shape[1]):
        resolution = 1024
        images = []
        images2 = []
        images3 = []
        for k in range(sample.shape[0]):
            draw = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
            draw2 = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw2.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
            draw3 = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw3.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
            draw_color = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw_color.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='white'))
            polys = []
            types = []
            for j, point in (enumerate(sample[k][i])):
                if model_kwargs[f'src_key_padding_mask'][i][j]==1:
                    continue
                point = point.cpu().data.numpy()
                if j==0:
                    poly = []
                if j>0 and (model_kwargs[f'room_indices'][i, j]!=model_kwargs[f'room_indices'][i, j-1]).any():
                    c = (len(polys)%28) + 1
                    polys.append(poly)
                    types.append(c)
                    poly = []
                pred_center = False
                if pred_center:
                    point = point/2 + 1
                    point = point * resolution//(2*2)#*2 added to fit whole puzzle on canvas
                else:
                    point = point/2 + 0.5
                    point = point * resolution//(1*2)#*2 added to fit whole puzzle on canvas
                poly.append((point[0], point[1]))
            c = (len(polys)%28) + 1
            polys.append(poly)
            types.append(c)
            for poly, c in zip(polys, types):
                room_type = c
                c = webcolors.hex_to_rgb(ID_COLOR[c])
                draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='black', stroke_width=1))
                draw.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill='black', fill_opacity=0.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                draw2.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                for corner in poly:
                    draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                    draw3.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
            #images.append(Image.open(io.BytesIO(cairosvg.svg2png(draw.asSvg()))))
            #images2.append(Image.open(io.BytesIO(cairosvg.svg2png(draw2.asSvg()))))
            #images3.append(Image.open(io.BytesIO(cairosvg.svg2png(draw3.asSvg()))))
            img = Image.open(io.BytesIO(cairosvg.svg2png(draw_color.as_svg())))
            if not nosave:
                if save_edges:
                    draw.save_svg(f'{output_path}/{ext}/{tmp_count+i}_{k}_{ext}.svg')
                if save_svg:
                    draw_color.save_svg(f'{output_path}/{ext}/{tmp_count+i}c_{k}_{ext}.svg')
                else:
                    img.save(f'{output_path}/{ext}/{tmp_count+i}c_{ext}.png')
        # if save_gif:
        #     imageio.mimwrite(f'outputs/gif/{tmp_count+i}.gif', images, fps=10, loop=1)
        #     imageio.mimwrite(f'outputs/gif/{tmp_count+i}_v2.gif', images2, fps=10, loop=1)
        #     imageio.mimwrite(f'outputs/gif/{tmp_count+i}_v3.gif', images3, fps=10, loop=1)
    return img