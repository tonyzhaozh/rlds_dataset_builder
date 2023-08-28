from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from aloha.conversion_utils import MultiThreadedDatasetBuilder
import h5py

import IPython
e = IPython.embed


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # print('\nMake new embed model\n')
    _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _parse_example(episode_path):
        # load raw data --> this should change for your dataset
        print(f'\n\nProccessing {episode_path}')
        # load raw data --> this should change for your dataset
        with h5py.File(episode_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]

            qpos = root['/observations/qpos'][:]
            # qvel = root['/observations/qvel'][:]
            image_dict = dict()
            for cam_name in ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][:]
            action = root['/action'][:]

        language = episode_path.split('/')[-2]
        print(language)

        dir_name_to_language = {'12_01_ziploc_slide_50': "Slide open the ziploc bag on the table.",
                                '1_22_cups_open': "Open the small condiment cup on the table.",
                                'aloha_bag_no_audio': "Put objects on the table into the transparent ziploc bag.",
                                'aloha_coffee_new': "Pick up the coffee pod on the table with left arm. Open the coffee maker with right arm. Put the coffee pod into the machine. Put the coffee mug on the drip tray. Press the buttons on top of the machine to start coffee making.",
                                'aloha_fork_pick_up': "Pick up the fork and put it inside the plate.",
                                'aloha_pingpong_test': "First pour one pingpong ball from right arm to left arm, then pour it back to the right arm.",
                                'aloha_screw_driver': "Pick up the yellow screw driver with right arm, hand it to the left arm, then drop it inside the cup.",
                                'aloha_towel': "Tear one segment of kitchen towel, and put it near the spilled yellow can.",
                                'battery': "Pick up the battery on the table, the insert it into the remote controller.",
                                'candy': "Unwrap the candy on the table by first twisting it and then peeling it open.",
                                'tape': "Cut a short segment of tape using the dispenser, hand it to the left hand and hang it at the edge of the cardboard box.",
                                'thread_velcro': "Pick up the black velcro cable tie on the table, then thread one end of it into the other.",
                                }

        language = dir_name_to_language.get(language, language)
        print(language)

        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(episode_len):
            # compute Kona language embedding
            if i == 0:
                # only run language embedding once since instruction is constant -- otherwise very slow
                language_embedding = _embed([language])[0].numpy()

            episode.append({
                'observation': {
                    'image': image_dict['cam_low'][i],
                    'image_top': image_dict['cam_high'][i],
                    'wrist_image_left': image_dict['cam_left_wrist'][i],
                    'wrist_image_right': image_dict['cam_right_wrist'][i],
                    'state': qpos[i],
                },
                'action': action[i],
                'discount': 1.0,
                'reward': float(i == (episode_len - 1)),
                'is_first': i == 0,
                'is_last': i == (episode_len - 1),
                'is_terminal': i == (episode_len - 1),
                'language_instruction': language,
                'language_embedding': language_embedding,
            })
        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        yield _parse_example(sample)


class Aloha(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 24             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 15  # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Aloha cam_low RGB observation.',
                        ),
                        'image_top': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Aloha cam_high RGB observation.',
                        ),
                        'wrist_image_left': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Aloha cam_left_wrist RGB observation.',
                        ),
                        'wrist_image_right': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Aloha cam_right_wrist RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float32,
                            doc='Robot state, consists of 14x robot joint angles, '
                                '(7 for each of the two arms.)',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float32,
                        doc='Robot action, consists of 14x absolute joint angles, '
                            '(7 for each of the two arms.)',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))
    def _split_paths(self):
        """Define filepaths for data splits."""
        paths = glob.glob('/scr/tonyzhao/rlds/*/episode*.hdf5')
        print(len(paths))

        return {
            'train': paths,
        }

