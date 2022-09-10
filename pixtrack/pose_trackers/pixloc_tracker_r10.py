import os
import numpy as np
import torch
from pathlib import Path

from pixloc.utils.colmap import Camera as ColCamera
from pixloc.pixlib.geometry import Camera as PixCamera, Pose
from pixloc.utils.data import Paths

from pixtrack.utils.pose_utils import geodesic_distance_for_rotations, get_camera_in_world_from_pixpose
from pixtrack.pose_trackers.base_pose_tracker import PoseTracker
from pixtrack.localization.pixloc_pose_refiners import PoseTrackerLocalizer
from pixtrack.localization.tracker import DebugTracker
from pixtrack.utils.io import ImageIterator
from pixtrack.utils.hloc_utils import extract_covisibility
from pixtrack.utils.ingp_utils import load_nerf2sfm, initialize_ingp, sfm_to_nerf_pose, nerf_to_sfm_pose
from pixtrack.visualization.run_vis_on_poses import get_nerf_rgb_image, get_nerf_depth_image
from scipy.spatial.transform import Rotation as R

import pycolmap
from pycolmap import infer_camera_from_image

import cv2
import pickle as pkl
import argparse

class PixLocPoseTrackerR9(PoseTracker):
    def __init__(self, data_path, loc_path, eval_path, debug=False):
        default_paths = Paths(
                            query_images='query/',
                            reference_images=loc_path,
                            reference_sfm='aug_sfm',
                            query_list='*_with_intrinsics.txt',
                            global_descriptors='features.h5',
                            retrieval_pairs='pairs_query.txt',
                            results='pixloc_object.txt',)
        conf = {
                'experiment': 'pixloc_megadepth',
                'features': {},
                'optimizer': {
                              'num_iters': 150,
                              'pad': 1,
                             },
                'refinement': {
                               'num_dbs': 1,
                               'multiscale': [4, 1],
                               'point_selection': 'all',
                               'normalize_descriptors': True,
                               'average_observations': False,
                               'do_pose_approximation': False,
                              },
                }
        self.debug = debug
        paths = default_paths.add_prefixes(Path(data_path),
                                           Path(loc_path),
                                           Path(eval_path))
        self.localizer = PoseTrackerLocalizer(paths, conf)
        self.eval_path = eval_path
        covis_path = Path(paths.reference_sfm) / 'covis.pkl'
        if os.path.isfile(covis_path):
            self.covis = pkl.load(open(covis_path, 'rb'))
        else:
            self.covis = extract_covisibility(paths.reference_sfm)
            with open(str(covis_path), 'wb') as f:
                pkl.dump(self.covis, f)
        self.pose_history = {}
        self.pose_tracker_history = {}
        self.cold_start = True
        self.pose = None
        upright_ref_img = os.environ['UPRIGHT_REF_IMG']
        self.reference_ids = [self.localizer.model3d.name2id[upright_ref_img]]
        nerf_path = Path(os.environ['SNAPSHOT_PATH']) / 'weights.msgpack'
        nerf2sfm_path = Path(os.environ['PIXSFM_DATASETS']) / os.environ['OBJECT'] / 'nerf2sfm.pkl'
        self.reference_scale = 1.0
        self.localizer.refiner.reference_scale = self.reference_scale
        self.nerf2sfm = load_nerf2sfm(str(nerf2sfm_path))
        self.testbed = initialize_ingp(str(nerf_path))
        self.dynamic_id = None
        self.hits = 0
        self.misses = 0
        self.cache_hit = False
        point_ids = []
        num_of_3d_points = len(self.localizer.model3d.points3D)
        points3d_in_nerf = []
        points3d_in_sfm = []
        for point_id, point in enumerate(self.localizer.model3d.points3D):
            point_ids.append(point)
            pose_placeholder = np.eye(4)
            pose_placeholder[:3, -1] = self.localizer.model3d.points3D[point].xyz
            points3d_in_sfm.append(self.localizer.model3d.points3D[point].xyz)
            pose_placeholder_nerf = sfm_to_nerf_pose(nerf2sfm=self.nerf2sfm, sfm_pose=pose_placeholder)
            points3d_in_nerf.append(pose_placeholder_nerf[:3, -1])
        with open("nerf_pts.npy", "wb") as f:
            np.save(f, points3d_in_nerf, allow_pickle=True, fix_imports=True)
        self.points3d_in_nerf = np.array(points3d_in_nerf)
        self.points3d_in_sfm = np.array(points3d_in_sfm)
        with open("sfm_pts.npy", "wb") as f:
            np.save(f, points3d_in_sfm, allow_pickle=True, fix_imports=True)
        self.point_ids = point_ids

    def relocalize(self, query_path):
        if self.cold_start:
            self.camera = self.get_query_camera(query_path)
            self.cold_start = False
        ref_img = self.localizer.model3d.dbs[self.reference_ids[0]]
        rotation = ref_img.qvec2rotmat()
        translation = ref_img.tvec
        pose_init = Pose.from_Rt(rotation,
                                 translation)
        self.pose = pose_init
        return

    def get_query_camera(self, query):
        camera = infer_camera_from_image(query)
        camera = ColCamera(None,
                        camera.model_name,
                        int(camera.width),
                        int(camera.height),
                        camera.params)
        camera = PixCamera.from_colmap(camera)
        return camera

    def update_reference_ids(self):
        if self.cache_hit == True:
            return self.reference_ids
        curr_refs = self.reference_ids
        curr_pose = self.pose
        R_qry = curr_pose.numpy()[0]
        cimg = self.localizer.model3d.dbs[curr_refs[0]]
        R_ref = cimg.qvec2rotmat()
        curr_gdist = geodesic_distance_for_rotations(R_qry, R_ref)
        covis = self.covis[curr_refs[0]]
        N = 50
        covis = {k: covis[k] for k in covis if covis[k] > N}
        gdists = {curr_refs[0]: curr_gdist}
        for ref in covis:
            cimg = self.localizer.model3d.dbs[ref]
            R_ref = cimg.qvec2rotmat()
            gdist = geodesic_distance_for_rotations(R_qry, R_ref)
            gdists[ref] = gdist

        reference_ids = sorted(gdists, key=lambda x: gdists[x])
        K = 1
        self.reference_ids = reference_ids[:K]
        return self.reference_ids

    def get_reference_image(self, pose):
        cIw = get_camera_in_world_from_pixpose(pose)
        nerf_pose = sfm_to_nerf_pose(self.nerf2sfm, cIw)
        ref_camera = self.localizer.model3d.cameras[1]
        ref_camera = PixCamera.from_colmap(ref_camera)
        ref_camera = ref_camera.scale(self.reference_scale)
        nerf_rgb_img = get_nerf_rgb_image(self.testbed, nerf_pose, ref_camera)
        nerf_depth_img = get_nerf_depth_image(self.testbed, nerf_pose, ref_camera)
        return nerf_rgb_img, nerf_depth_img

    def create_dynamic_reference_image(self, pose):
        rgb_img, depth_img = self.get_reference_image(pose)
        dynamic_id = hash(str(np.vstack((pose.numpy()[0], pose.numpy()[1]))))
        features = self.localizer.refiner.extract_reference_features(self.reference_ids, pose, rgb_img)
        camera_pose_in_sfm = np.eye(4)
        camera_pose_in_sfm[:3, :3] = pose.numpy()[0]
        camera_pose_in_sfm[:3, -1] = pose.numpy()[1]
        points3d_in_nerf = np.hstack((self.points3d_in_nerf, np.ones((self.points3d_in_nerf.shape[0], 1))))
        nerf_camera_pose = sfm_to_nerf_pose(self.nerf2sfm, camera_pose_in_sfm)
        points3d_in_nerf_camera_frame = np.dot(
            sfm_to_nerf_pose(self.nerf2sfm, camera_pose_in_sfm), points3d_in_nerf.T
        ).T
        with open("nerf_camera_pts.npy", "wb") as f:
            np.save(f, points3d_in_nerf_camera_frame, allow_pickle=True, fix_imports=True)
        points3d_in_sfm = np.hstack((self.points3d_in_sfm, np.ones((self.points3d_in_sfm.shape[0], 1))))
        points3d_in_sfm_camera_frame = np.dot(
            camera_pose_in_sfm, points3d_in_sfm.T
        ).T
        with open("sfm_camera_pts.npy", "wb") as f:
            np.save(f, points3d_in_sfm_camera_frame, allow_pickle=True, fix_imports=True)
        points3d_in_nerf_camera_frame_new = []
        for point_id, point in enumerate(points3d_in_sfm_camera_frame):
            pose_placeholder = np.eye(4)
            pose_placeholder[:3, -1] = point[:3]
            point = sfm_to_nerf_pose(self.nerf2sfm, pose_placeholder)[:3, -1]
            points3d_in_nerf_camera_frame_new.append(point)
        with open("nerf_camera_pts_new.npy", "wb") as f:
            np.save(f, points3d_in_nerf_camera_frame_new, allow_pickle=True, fix_imports=True)

        ref_camera = self.localizer.model3d.cameras[1]
        f, cy, cx, d  = ref_camera.params.tolist()
    
        ref_camera = PixCamera.from_colmap(ref_camera)
        ref_camera = ref_camera.scale(self.reference_scale)
        pts_in_image = ref_camera.world2image(points3d_in_sfm_camera_frame[:, :-1])
        # Make one pixel red
        viz_img = rgb_img.copy()
        viz_img_depth = np.uint8(depth_img.copy() * 255)
        count = 0
        points_from_depth = []
        for num, pt in enumerate(pts_in_image[0]):
            try:
                depth_at_point = depth_img[int(pt[1]), int(pt[0])]
            except IndexError:
                continue

            if depth_at_point[0] == 0.0:
                #print("ignored as depth was zero")
                continue
            if depth_at_point[-1] < 0.3:
                count += 1
                continue
            point_from_depth = [(pt[0].item() - cx) * depth_at_point[0] / f, (pt[1].item() - cy) * depth_at_point[0] / f, depth_at_point[0]]
            points_from_depth.append(point_from_depth)
            point3d = points3d_in_nerf_camera_frame[num]
            point3d_new = points3d_in_nerf_camera_frame_new[num]
            if -point3d_new[0] <= depth_at_point[0] or np.isclose(-point3d_new[0], depth_at_point[0], rtol=0.05):
                viz_img = cv2.circle(viz_img, center=(int(pt[0]),int(pt[1])), radius=1, color=(0, 0, 255), thickness=-1)
                viz_img_depth = cv2.circle(viz_img_depth, center=(int(pt[0]),int(pt[1])), radius=2, color=(0, 0, 255), thickness=-1)
                pass
            else:
                viz_img = cv2.circle(viz_img, center=(int(pt[0]),int(pt[1])), radius=1, color=(0, 255, 0), thickness=-1)
                #viz_img_depth = cv2.circle(viz_img_depth, center=(int(pt[0]),int(pt[1])), radius=2, color=(0, 0, 255), thickness=-1)
            #print(depth_at_point, point3d, point3d_new)
        # Save
        with open("points_from_depth.npy", "wb") as f:
            np.save(f, np.array(points_from_depth), allow_pickle=True, fix_imports=True)
        points_from_depth_nerf = []
        for point_id, point in enumerate(points_from_depth):
            pose_placeholder = np.eye(4)
            pose_placeholder[:3, -1] = point[:3]
            point = sfm_to_nerf_pose(self.nerf2sfm, pose_placeholder)[:3, -1]
            points_from_depth_nerf.append(point)
        with open("points_from_depth_nerf.npy", "wb") as f:
            np.save(f, np.array(points_from_depth_nerf), allow_pickle=True, fix_imports=True)
        points_from_depth_sfm = []
        for point_id, point in enumerate(points_from_depth):
            pose_placeholder = np.eye(4)
            pose_placeholder[:3, -1] = point[:3]
            point = nerf_to_sfm_pose(self.nerf2sfm, pose_placeholder)[:3, -1]
            points_from_depth_sfm.append(point)
        with open("points_from_depth_sfm.npy", "wb") as f:
            np.save(f, np.array(points_from_depth_nerf), allow_pickle=True, fix_imports=True)
        print(count, "num")
        cv2.imwrite("/home/ubuntu/result.png",viz_img)
        cv2.imwrite("/home/ubuntu/result_depth.png",viz_img_depth)
        return dynamic_id, features

    def get_dynamic_id(self, pose):
        features_dicts = self.localizer.refiner.features_dicts
        if self.dynamic_id is None:
            self.dynamic_id, features = self.create_dynamic_reference_image(self.pose)
            features_dicts[self.dynamic_id] = {}
            features_dicts[self.dynamic_id]['pose'] = self.pose
            features_dicts[self.dynamic_id]['features'] = features
            return self.dynamic_id

        curr_pose = self.pose
        R_qry = curr_pose.numpy()[0]
        dynamic_pose = features_dicts[self.dynamic_id]['pose']
        R_drf = dynamic_pose.numpy()[0]
        self.cache_hit = False
        self.dynamic_id, features = self.create_dynamic_reference_image(self.pose)
        features_dicts[self.dynamic_id] = {}
        features_dicts[self.dynamic_id]['pose'] = self.pose
        features_dicts[self.dynamic_id]['features'] = features
        features_dicts[self.dynamic_id]['ref_ids'] = self.update_reference_ids()
        self.misses += 1
        self.cache_hit = True

        return self.dynamic_id


    def refine(self, query):
        query_path, query_image = query
        if self.cold_start:
            self.relocalize(query_path)
            self.cold_start = False

        self.dynamic_id = self.get_dynamic_id(self.pose)
        translation = self.pose.numpy()[1]
        rotation = self.pose.numpy()[0]
        rot = R.from_matrix(rotation)
        rotation = rot.as_matrix()
        trackers = {}
        rets = {}
        costs = {}
        for ref_id in self.reference_ids:
            ref_img = self.localizer.model3d.dbs[ref_id]
            pose_init = Pose.from_Rt(rotation, translation)
            tracker = DebugTracker(self.localizer.refiner, self.debug)
            ret = self.localizer.run_query(
                query_path,
                self.camera,
                pose_init,
                [ref_id],
                image_query=query_image,
                pose=self.pose,
                reference_images_raw=None,
                dynamic_id=self.dynamic_id
            )
            rets[ref_id] = ret
            trackers[ref_id] = tracker
            avg_cost = np.mean([x[-1] for x in tracker.costs])
            costs[ref_id] = avg_cost
        best_ref_id = min(costs, key=costs.get)
        ret = rets[best_ref_id]
        success = ret['success']
        if success:
            self.pose = ret['T_refined']
        ret['camera'] = self.camera
        ret['reference_ids'] = self.reference_ids
        ret['query_path'] = query_path
        img_name = os.path.basename(query_path)
        self.pose_history[img_name] = ret
        self.pose_tracker_history[img_name] = trackers[best_ref_id]
        return success

    def get_query_frame_iterator(self, image_folder, max_frames):
        iterator = ImageIterator(image_folder, max_frames)
        return iterator

    def save_poses(self):
        path = os.path.join(self.eval_path, 'poses.pkl')
        with open(path, 'wb') as f:
            pkl.dump(self.pose_history, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', default='IMG_4117')
    parser.add_argument('--out_dir', default='IMG_4117')
    parser.add_argument('--frames', type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    obj = os.environ['OBJECT']
    data_path = Path(os.environ['PIXSFM_DATASETS']) / obj
    eval_path = Path(args.out_dir)
    loc_path = Path(os.environ['PIXTRACK_OUTPUTS']) / 'nerf_sfm' / ('aug_%s' % obj)
    if not os.path.isdir(eval_path):
        os.makedirs(eval_path)
    tracker = PixLocPoseTrackerR9(data_path=str(data_path),
                                  eval_path=str(eval_path),
                                  loc_path=str(loc_path),
                                  debug=args.debug)
    tracker.run(args.query, max_frames=args.frames)
    tracker.save_poses()
    print('Cache hits: %d, misses: %d' % (tracker.hits, tracker.misses))
    tracker_path = os.path.join(tracker.eval_path, 'trackers.pkl')
    with open(tracker_path, 'wb') as f:
        pkl.dump(tracker.pose_tracker_history, f)

    print('Done')
