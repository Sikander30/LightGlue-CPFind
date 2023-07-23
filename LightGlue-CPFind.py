import torch.cuda
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from nvjpeg import NvJpeg
import argparse
import os
import time


def find_points(input_file_path: str, output_file_path: str, working_dir, extractor, matcher, batch_size=16):
    images = []

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:

            for line in input_file:
                output_file.write(line)

                if line[0] == 'i':
                    fname_start = line.find('n')
                    if fname_start != -1:
                        fname = line[fname_start: -1].split('"')
                        images.append(fname[1])

                # From now on we have to write the control points
                if '# control points\n' in line:
                    break

            # Extract keypoints
            features = []
            num_images = len(images)
            nj = NvJpeg()
            for current_image in range(num_images):
                print(f'Extracting features from {images[current_image]}.')
                # image = load_image(os.path.join(working_dir, images[current_image])).cuda()
                # nvjpg is much faster
                image = (torch.from_numpy(
                    nj.read(os.path.join(working_dir, images[current_image]))).float().cuda() / 255.0).permute(2, 0, 1)
                feats = extractor.extract(image)
                features.append(
                    {
                        'keypoints': feats['keypoints'].detach(),
                        'keypoint_scores': feats['keypoint_scores'].detach(),
                        'descriptors': feats['descriptors'].detach(),
                        'image_size': feats['image_size'].detach()
                    }
                )

            # Find matches
            control_points = []
            for i in range(num_images):
                for j in range(i - 1):
                    matches = rbd(matcher({'image0': features[i], 'image1': features[j]}))['matches'].detach().cpu()
                    kpts0 = rbd(features[i])['keypoints'].detach().cpu()
                    kpts1 = rbd(features[j])['keypoints'].detach().cpu()
                    m_kpts0 = kpts0[matches[..., 0]]
                    m_kpts1 = kpts1[matches[..., 1]]

                    n_matches = len(matches)
                    for k in range(n_matches):
                        control_points.append({
                            'image': i,
                            'IMAGE': j,
                            'x': m_kpts0[k, 0],
                            'y': m_kpts0[k, 1],
                            'X': m_kpts1[k, 0],
                            'Y': m_kpts1[k, 1]
                        })

            print(f'Found {len(control_points)} control points.')

            # Write control points to project file
            for pnt in control_points:
                output_file.write(
                    f'c n{pnt["image"]} N{pnt["IMAGE"]} x{pnt["x"]} y{pnt["y"]} X{pnt["X"]} Y{pnt["Y"]} t0\n')

            # Finish copying file
            for line in input_file:
                if line[0] != 'c':  # Do not copy old control points
                    output_file.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LightGlue-CPFind',
        description='Finds control points in image pairs using LightGlue',
    )

    parser.add_argument('input_project')
    parser.add_argument('-o', '--output', help='Output file.', required=True)

    args = parser.parse_args()

    # Probably useless
    if args.input_project is None:
        print('LightGlue-CPFind: No project file given')
        exit(-1)

    extractor = SuperPoint(max_num_keypoints=128).eval().cuda()
    matcher = LightGlue(features='superpoint').eval().cuda()

    project_path = os.path.abspath(args.input_project)
    working_dir = os.path.dirname(project_path)

    out_path = f'{args.output}.temp'

    start_time = time.time()
    find_points(args.input_project, out_path, working_dir, extractor, matcher)
    end_time = time.time()
    print(f'Execution time: {end_time - start_time} seconds.')

    os.replace(out_path, args.output)
