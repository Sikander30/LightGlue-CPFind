# The present Work is distributed under the terms of the Apache License Version 2.0.
# You should have received a copy of the aforementioned license along with the present file.
# If that is not che case the full text of the license is available at : https://www.apache.org/licenses/LICENSE-2.0.txt
# The Work is provided "AS-IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied, including, without limitation, any warranties or conditions
# of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
# You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any
# risks associated with Your exercise of permissions under this License.

import torch
import torchvision
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import rbd
import argparse
import os
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


def find_points(input_file_path: str, output_file_path: str, work_dir, feat_extractor, feat_matcher):
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
            for current_image in range(num_images):
                print(f'Extracting features from {images[current_image]}.')
                image = torchvision.io.read_image(os.path.join(work_dir,
                                                               images[current_image])).to(device,
                                                                                          dtype=torch.float32) / 255.0
                feats = feat_extractor.extract(image)
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
                    matches = rbd(feat_matcher({'image0': features[i], 'image1': features[j]}))[
                        'matches'].detach().cpu()
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
    parser.add_argument('-f', '--features',
                        help='Feature extractor (superpoint or disk, default: superpoint)', required=False)
    parser.add_argument('-m', '--max-keypoints',
                        help='Maximum number of keypoints to extract from each image (default: 128)', required=False)

    args = parser.parse_args()

    # Probably useless
    if args.input_project is None:
        print('LightGlue-CPFind: No project file given')
        exit(-1)

    extractor_name = 'superpoint'
    if args.features is not None:
        if args.features.lower in ('superpoint', 'disk'):
            extractor_name = args.features.lower
        else:
            print(f'ERROR: unknown feature extractor {args.features}.')
            exit(-1)

    max_kpts = 128
    if args.max_keypoints is not None:
        max_kpts = int(args.max_keypoints)

    extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(device) if extractor_name == 'superpoint' else DISK(
        max_num_keypoints=max_kpts).eval().to(device)
    # extractor = DISK(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features=extractor_name).eval().to(device)

    project_path = os.path.abspath(args.input_project)
    working_dir = os.path.dirname(project_path)

    out_path = f'{args.output}.temp'

    start_time = time.time()
    find_points(args.input_project, out_path, working_dir, extractor, matcher)
    end_time = time.time()
    print(f'Execution time: {end_time - start_time} seconds.')

    os.replace(out_path, args.output)
