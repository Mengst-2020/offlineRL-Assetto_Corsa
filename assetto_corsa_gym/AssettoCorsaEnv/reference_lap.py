import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
# from AssettoCorsaEnv.curvature import curvature_splines
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

def curvature_splines(x, y, error=0.1):
    """Calculate the signed curvature of a 2D curve at each point
    using interpolating splines.
    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )
         or
         y=None and
         x is a numpy.array(dtype=complex) shape (n_points, )
         In the second case the curve is represented as a np.array
         of complex numbers.
    error : float
        The admisible error when interpolating the splines
    Returns
    -------
    curvature: numpy.array shape (n_points, )
    Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
    but more accurate, especially at the borders.
    """
    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

    xˈ = fx.derivative(1)(t)
    xˈˈ = fx.derivative(2)(t)
    yˈ = fy.derivative(1)(t)
    yˈˈ = fy.derivative(2)(t)
    curvature = (xˈ* yˈˈ - yˈ* xˈˈ) / np.power(xˈ** 2 + yˈ** 2, 3 / 2)
    return curvature


import logging
logger = logging.getLogger(__name__)

def get_yaw(x,  y):
    deriv_x = np.diff(x)
    deriv_z = np.diff(y)
    angle = np.arctan2(deriv_z, deriv_x)
    # do not wrap, we want the angle from -pi to pi
    return angle

def distSegment2Index(rl_dist, l_bound, u_bound):
    return np.where(np.logical_and(rl_dist >= l_bound, rl_dist <= u_bound))

def calculate_distance_from_xy(x, y):
    x_diff = np.diff(x)
    y_diff = np.diff(y)

    dist = np.cumsum(np.hypot(x_diff, y_diff) )
    dist = np.insert(dist, 0, 0)
    return dist

def convert_to_distance(distance, ch, interp_type="linear"):
    """
    Interpolate every channel to make the distance the independent variable

    interp_type: "linear" or cubic (linear x6 faster)
    """
    def monotonic(x):
        dx = np.diff(x)
        #print(np.all(dx >= 0))
        return np.all(dx <= 0) or np.all(dx >= 0), dx

    _, dx = monotonic(distance)

    # check for consecutive equal samples
    # for the interpolation we need a monotonically increasing function
    equal_samples = np.where(dx <= 0)[0]
    if len(equal_samples):
        print("Warning: check for consecutive equal samples. Found ", equal_samples, "samples")
        last_sample = equal_samples[0]# - 1
        print("Will crop from %d samples to %d samples" % (len(distance), last_sample))
    else:
        last_sample = -1

    distance = distance[:last_sample]
    ch = ch[:last_sample]
    tck = {}
    out = {}
    p = distance
    last_dist = distance[-1] + 3 # add 3 meters to the interpolation
    x = np.arange(0, last_dist)
    tck = interp1d(p, ch, kind=interp_type, fill_value="extrapolate")
    return tck(x)

class ReferenceLap:
    """
    Can use any line. Just x,y are needed. Distance, yaw and curvature are calculated
    Optionally curvature could be from file and target_speed

    Warning: the last point of the racing line should precede the first point to avoid discontinuities in the gap.

    racing_line
        array:
            pos,xyz, dist, yaw, curvature

    racing_line_dist
        array
            same but interpolated and cropped so that the distance is monotonically increasing
    """
    file_channels = ["pos_x", "pos_y"]
    target_speed_channel_name = "target_speed"

    # new names (access with .index("pos_x"))
    # channels =      ["position.x", "position.y", "lapDistance", "yaw", "curvature", "target_speed"]
    # channels_dist = ["position.x", "position.y", "lapDistance", "yaw", "curvature", "target_speed"]
    channels =      ["position.x", "position.y", "lapDistance", "yaw", "curvature"]
    channels_dist = ["position.x", "position.y", "lapDistance", "yaw", "curvature"]

    SEG_TYPE = {
        "straight": 0,
        "hairpin": 1,
        "short s": 2,
        "long s": 3,
        "double_apex": 4,
        "sweeper": 5,
        "medium-corner":6,
        } 
    INV_SEG_TYPE={v:k for k,v in SEG_TYPE.items()}

    def __init__(self, file_path, use_target_speed):
        logger.info(f"Reference Lap. Loading: {file_path}")
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.df = self.df.reset_index()
        self.use_target_speed = use_target_speed
        self.cur_idx = 0
        self.n = 0
        self.in_out=False

        try:
            self.ts = self.df[self.file_channels].values
        except KeyError:
            logger.error(f"Channels {self.file_channels} not found.")
            logger.error(f"Channels in racing line file: {self.df.columns}")
            raise

        if self.use_target_speed:
            self.target_speed = self.df[self.target_speed_channel_name].values.reshape(-1,1)
            print(self.target_speed)
            logger.info("Using target speed")
            self.channels.append(self.target_speed_channel_name)
            self.channels_dist.append(self.target_speed_channel_name)

        # calculate distance channels from x,y coordinates
        self.distance_ch_time = calculate_distance_from_xy(self.ts[:,0], self.ts[:,1])
        self.ts = np.concatenate([self.ts, self.distance_ch_time.reshape(-1,1)], axis=1)

        # calculate angle_y from x,y coordinates
        # calculate the yaw, fl gives a yaw wrapped to pi
        yaw = get_yaw(self.ts[:,0], self.ts[:,1])
        yaw = np.insert(yaw, 0, yaw[0]).reshape(-1,1)
        self.ts = np.concatenate([self.ts, yaw], axis=1)

        # calculate curvature
        # If the curvature is present in the racing line use it else calculate it
        if "curvature" in self.df.columns:
            logger.info("Using curvature from racing line file")
            curvatures = self.df["curvature"].values.reshape(-1,1)
        else:
            logger.info("Calculating curvature")
            curvatures = curvature_splines( self.ts[:,0],  self.ts[:,1] )
            curvatures = curvatures.reshape(-1,1)
        self.ts = np.concatenate([self.ts, curvatures], axis=1)

        if self.use_target_speed:
            self.ts = np.concatenate([self.ts, self.target_speed], axis=1)

        # interpolate to distance
        td = []
        for _, ch in enumerate(self.channels_dist):
            idx = self.channels.index(ch)
            td.append( convert_to_distance(self.distance_ch_time, self.ts[:,idx]) )
        self.td = np.array( td ).T
        self.distance_ch_dist = self.td[:,2]

        res=self.classify_corners_from_curvature()
        lapdist=self.ts[:,2]
        segments = []   # [(lapdist_start, lapdist_end), ...]
        labels   = []   # ["hairpin", "chicane/S", ...]
        for r in res:
            seg = (float(lapdist[r["start"]]), float(lapdist[r["end"]]))
            segments.append(seg)
            labels.append(r["label"])
        self.segments = np.array(segments)   # shape: (N, 2)
        self.labels   = np.array(labels)
        self.type = np.array([self.INV_SEG_TYPE[int(i)] for i in self.labels])
        self.n=len(self.segments)

    def get_racing_line_time(self):
        return self.ts[:, 0:2]

    def get_racing_line_dist(self):
        return self.td[:, 0:2]

    def get_channel_time(self, channel_name):
        return self.ts[:,self.channels.index(channel_name)].reshape(-1,1)

    def get_channel_dist(self, channel_name):
        return self.td[:,self.channels.index(channel_name)].reshape(-1,1)

    """
    Curvature look ahead
    """
    def distSegment2Index(self, rl_dist, l_bound, u_bound):
        return np.where((rl_dist >= l_bound) & (rl_dist <= u_bound))[0]

    def getLADVector(self, rl_dist, dist, LA_dist, vector_size, channel):
        """
            Get a vector (len vector_size) of a channel at max LA_dist

            rl_dist: time series with the distance channel interpolated and projected to distance
            dist: current distance of the car
            LA_dist: how far to look ahead [m]
            vector_size: downsample the result to this value
            channel: distance interpolated channel
            returns: vector of vector_size with the channel interpolated by distance

        """
        rl_dist = rl_dist.copy()
        patch = 0
        track_len = rl_dist[-1]

        if ((dist - track_len) > 50):
            print("## look ahead was out of range!!! Will return a Zero Vector", dist, track_len)
            assert ((dist - track_len) > 50), "distance was more than 50 meters bigger than the track len dist %f track_len %f" \
                                               % (dist, track_len)

        start = dist
        end = dist + LA_dist
        segment = self.distSegment2Index(rl_dist, start, end)

        if end > track_len:
            patch = end - track_len
            segment = np.concatenate( [segment, self.distSegment2Index(rl_dist, 0, patch)] )

        vector = channel[segment]
        vector = vector[0::len(vector) // vector_size]
        vector = vector[0:vector_size]
        return vector, segment, patch

    def get_curvature_segment(self, dist, LA_dist, vector_size):
        """
        Get single value curvature:
            dist: starting distance in the racing line
            LA_dist: lookahed starting from dist
            vector_size: downsampled signal size
        """
        curv_index = self.channels_dist.index("curvature")
        vector, segment, patch = self.getLADVector(self.distance_ch_dist, dist, LA_dist, vector_size, self.td[:, curv_index])
        return vector

    def get_target_speed_segment(self, dist, LA_dist, vector_size):
        """
        Get single value curvature:
            dist: starting distance in the racing line
            LA_dist: lookahed starting from dist
            vector_size: downsampled signal size
        """
        assert self.use_target_speed, "target speed not used"

        target_speed_index = self.channels_dist.index("target_speed")
        vector, segment, patch = self.getLADVector(self.distance_ch_dist, dist, LA_dist, vector_size, self.td[:, target_speed_index])
        return vector

    def get_target_speed_value(self, dist):
        """
        Get single value of the target speed
        """
        assert self.use_target_speed, "target speed not enabled"

        target_speed_index = self.channels_dist.index("target_speed")
        vector, segment, patch = self.getLADVector(self.distance_ch_dist, dist, 200., 1, self.td[:, target_speed_index])
        return vector[0]

    def get_yaw(self, dist):
        """
        Get single value curvature
        """
        curv_index = self.channels_dist.index("yaw")
        vector, segment, patch = self.getLADVector(self.distance_ch_dist, dist, 200., 1, self.td[:,curv_index])
        return vector[0]

    def get_curvature(self, dist):
        """
        Get single value curvature
        """
        curv_index = self.channels_dist.index("curvature")
        vector, segment, patch = self.getLADVector(self.distance_ch_dist, dist, 200., 1, self.td[:,curv_index])
        return vector

    def cropped_racing_line(self, start, segment_len, vector_len):
        pos_x_idx = self.channels_dist.index("pos_x")
        pos_y_idx = self.channels_dist.index("pos_y")
        racing_line_cropped_x, _, _ = self.getLADVector(self.distance_ch_dist, start,
                                                        segment_len, vector_len, self.td[:,pos_x_idx])
        racing_line_cropped_y, _, _ = self.getLADVector(self.distance_ch_dist, start,
                                                        segment_len, vector_len, self.td[:,pos_y_idx])
        return np.vstack([ racing_line_cropped_x, racing_line_cropped_y ]).T

    def smooth_moving_avg(self, x, win= 21):
        if win < 3 or win % 2 == 0:
            return x.copy()
        pad = win // 2
        xpad = np.pad(x, (pad, pad), mode="edge")
        ker = np.ones(win) / win
        return np.convolve(xpad, ker, mode="valid")

    def _boolean_close(self, mask, close_win=5):
        if close_win <= 1: 
            return mask
        pad = close_win // 2
        # dilation
        dil = np.copy(mask)
        for i in range(len(mask)):
            lo = max(0, i-pad); hi = min(len(mask), i+pad+1)
            dil[i] = np.any(mask[lo:hi])
        # erosion
        ero = np.copy(dil)
        for i in range(len(mask)):
            lo = max(0, i-pad); hi = min(len(mask), i+pad+1)
            ero[i] = np.all(dil[lo:hi])
        return ero

    def _find_regions(self, mask):
        regs=[]; in_run=False; s=0
        for i,v in enumerate(mask):
            if v and not in_run:
                in_run=True; s=i
            elif not v and in_run:
                regs.append((s,i)); in_run=False
        if in_run: regs.append((s,len(mask)))
        return regs

    def _peak_indices(self, x, min_prom):
        peaks=[]
        for i in range(1, len(x)-1):
            if abs(x[i])>abs(x[i-1]) and abs(x[i])>abs(x[i+1]) and abs(x[i])>=min_prom:
                peaks.append(i)
        return peaks

    def classify_corners_from_curvature(
        self,
        smooth_win= 31,     #移动平均窗口
        cur_pts= 0.21,      #弯段曲率阈值
        k_eps = None,
        min_len_pts= 60,    #最小弯段长度
        close_win= 17,      #弯道链接
        hairpin_pct= 0.90,  #发卡阈值
        highspeed_pct= 0.80,#长弯阈值
        chicane_gap_pts= 60,#S弯长度阈值
    ):
        """Return list of segments with labels.
        labels ∈ {'hairpin','chicane/S','double-apex','high-speed-sweeper','medium-corner'}"""
        curvatures=self.ts[:,-1]
        x=self.ts[:,0]
        y=self.ts[:,1]
        k = curvatures.astype(float)
        k_s = self.smooth_moving_avg(k, smooth_win)
        # keep length consistent with input
        pad = (len(k)-len(k_s))//2
        if pad>0: k_s = np.pad(k_s, (pad, len(k)-len(k_s)-pad), mode='edge')

        absk = np.abs(k_s)
        if k_eps is None:
            k_eps = np.percentile(absk,90)*cur_pts

        k_hairpin   = np.percentile(absk, hairpin_pct*100)          # 发卡阈值（大曲率）
        k_highspeed = np.percentile(absk[absk>0], highspeed_pct*100) if np.any(absk>0) else 0.0

        corner_mask = absk > k_eps
        corner_mask = self._boolean_close(corner_mask, close_win)
        regs = [r for r in self._find_regions(corner_mask) if (r[1]-r[0]) >= min_len_pts]

        results=[]
        for (a,b) in regs:
            seg = k_s[a:b]; seg_abs = absk[a:b]
            L = (b-a)
            sign_changes = int(np.sum(np.diff(np.sign(seg)) != 0))
            kmax  = float(seg_abs.max()); kmean = float(seg_abs.mean())

            pk_idx = self._peak_indices(seg, min_prom=np.percentile(seg_abs,70))
            pk_signs = [np.sign(seg[i]) for i in pk_idx]
            pos_peaks = [i for i,s in zip(pk_idx, pk_signs) if s>0]
            neg_peaks = [i for i,s in zip(pk_idx, pk_signs) if s<0]

            # same-sign double apex?
            same_sign_double = any(np.sign(seg[pk_idx[i]]) == np.sign(seg[pk_idx[i+1]])
                                for i in range(len(pk_idx)-1))
            
            angles_in = np.arctan2(y[a+3]-y[a],x[a+3]-x[a])
            angles_out = np.arctan2(y[b]- y[b-3],x[b]-x[b-3])
            angle_diff = np.arctan2(np.sin(angles_out - angles_in), np.cos(angles_out - angles_in))
            angle_diff_deg = np.degrees(angle_diff)
            # --- rules ---
            label = 6
            if sign_changes >= 1 and len(pos_peaks)>=1 and len(neg_peaks)>=1:
                seg_sign = np.sign(seg)
                zero_idx = np.where(np.diff(seg_sign) != 0)[0] + 1 
                bounds = [0] + zero_idx.tolist() + [L-1]
                maxdelta_deg=0
                for si, ei in zip(bounds[:-1], bounds[1:]):
                    start_i = a + si
                    end_i   = a + ei
                    if end_i - start_i < 5:
                        continue  # 太短忽略，防止数值噪声

                    ang_in_sub = np.arctan2(y[start_i+3]-y[start_i],
                                            x[start_i+3]-x[start_i])
                    ang_out_sub = np.arctan2(y[end_i]-y[end_i-3],
                                             x[end_i]-x[end_i-3])
                    dtheta_sub = np.arctan2(np.sin(ang_out_sub-ang_in_sub),
                                            np.cos(ang_out_sub-ang_in_sub))
                    if maxdelta_deg<abs(dtheta_sub):
                        maxdelta_deg=abs(dtheta_sub)
                if abs(pos_peaks[0]-neg_peaks[0]) >= chicane_gap_pts:
                    if float(np.degrees(maxdelta_deg))<60:
                        label = 5
                    else:
                        label = 3
                else:
                    label = 2
            elif kmax >= k_hairpin and abs(angle_diff_deg) > 130 :
                label = 1
            elif (L >= 3*min_len_pts or abs(angle_diff_deg) < 45) :
                label = 5
            elif results:
                lastres=results[-1]
                if lastres["label"]==6 and (a-lastres["end"])<100 and np.sign(lastres["features"]["diff_deg"])==np.sign(angle_diff_deg):
                    a=lastres["start"]
                    label=4
                    sign_changes=1
                    results.pop()
            
            results.append({
                "start": int(a), "end": int(b), "label": label,
                "features": {"length_pts": int(b-a), "kmax": kmax, "kmean": kmean,
                            "sign_changes": sign_changes,
                            "k_eps": float(k_eps),
                            "diff_deg": float(angle_diff_deg)
                            }
            })
        return results

    def reset_num(self, lapdist_begin):
        if lapdist_begin >= self.segments[self.n-1][1]:
            self.in_out=False
            self.cur_idx=self.n
        else:
            for idx, (seg_start, seg_end) in enumerate(self.segments):
                if  lapdist_begin <= seg_start:
                    self.in_out=False
                    break
                elif lapdist_begin <= seg_end:
                    self.in_out=True
                    break
            self.cur_idx=idx

    def update_num(self, lapdist):
        if self.cur_idx < self.n:
            if self.in_out:
                if lapdist < self.segments[self.cur_idx][1]:    #弯中
                    return self.labels[self.cur_idx]
                else:                                           #出弯
                    self.cur_idx=self.cur_idx+1
                    self.in_out=False
                    return 0
            else:
                if lapdist < self.segments[self.cur_idx][0]:    #入弯前
                    return 0
                else:                                           #入弯
                    self.in_out=True
                    return self.labels[self.cur_idx]
        else:
            if lapdist<self.segments[0][0]:
                self.cur_idx=0
            return 0

    def diagnose_friction_failure(self, segments: pd.DataFrame, track_segments: pd.DataFrame,):
        lx = track_segments["left_border_x"].to_numpy()
        ly = track_segments["left_border_y"].to_numpy()
        rx = track_segments["right_border_x"].to_numpy()
        ry = track_segments["right_border_y"].to_numpy()

        cx = 0.5 * (rx + lx)
        cy = 0.5 * (ry + ly)

        ds = np.concatenate(([0.0], np.hypot(np.diff(cx), np.diff(cy))))
        s = np.cumsum(ds)

        # 曲率计算
        dx = np.gradient(cx) / np.maximum(np.gradient(s), 1e-6)
        dy = np.gradient(cy) / np.maximum(np.gradient(s), 1e-6)
        ddx = np.gradient(dx) / np.maximum(np.gradient(s), 1e-6)
        ddy = np.gradient(dy) / np.maximum(np.gradient(s), 1e-6)

        kappa = (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-6)
        abs_kappa = np.abs(kappa)

        # --- 冲出点 ---
        x_fail = segments["world_position_x"].to_numpy()[-1]
        y_fail = segments["world_position_y"].to_numpy()[-1]

        # 冲出点到中心线最近点
        dist2_center = np.hypot(cx - x_fail, cy - y_fail)
        idx_fail_center = int(np.argmin(dist2_center))

        s_fail_center = s[idx_fail_center]

        # --- 每个车辆采样对应的中心线索引 ---
        Nseg = len(segments)
        idx_center_for_seg = np.zeros(Nseg, dtype=int)

        seg_x = segments["world_position_x"].to_numpy()
        seg_y = segments["world_position_y"].to_numpy()

        for i in range(Nseg):
            dx_i = cx - seg_x[i]
            dy_i = cy - seg_y[i]
            dist2_i = dx_i**2 + dy_i**2
            idx_center_for_seg[i] = int(np.argmin(dist2_i))

        # --- 几何窗口与直道判定 ---
        L_back = 20.0       # 向以前  m
        L_front= 10.0
        L_straight = 80.0   # 较长直道长度
        thr_straight = 0.0025
        N_spike_max = 5     # 连续高曲率点数大于这个，认为不再是直道

        # 后向（冲出点前 -> 入弯）
        s_back = s_fail_center - L_back
        s_back_win_min = s_back - L_straight
        s_back_win_max = s_back

        mask_win_back = (s >= s_back_win_min) & (s <= s_back_win_max)
        kappa_win_back = abs_kappa[mask_win_back]

        high = kappa_win_back > thr_straight        
        h = high.astype(int)  
        is_back_long_straight = True                    
        if np.any(h):
            d = np.diff(np.concatenate(([0], h, [0])))
            run_starts = np.where(d == 1)[0]
            run_ends   = np.where(d == -1)[0] - 1
            run_lens   = run_ends - run_starts + 1

            if run_lens.size > 0 and np.max(run_lens) > N_spike_max:
                is_back_long_straight = False


        # 前向（冲出点后 -> 出弯）
        s_front_win_min = s_fail_center + L_front
        s_front_win_max = s_front_win_min + L_straight

        mask_win_front = (s >= s_front_win_min) & (s <= s_front_win_max)
        kappa_win_front = abs_kappa[mask_win_front]

        high = kappa_win_front > thr_straight        # bool
        h = high.astype(int)                        # 变成 0/1
        is_front_long_straight = True
        if np.any(h):
            d = np.diff(np.concatenate(([0], h, [0])))
            run_starts = np.where(d == 1)[0]
            run_ends   = np.where(d == -1)[0] - 1
            run_lens   = run_ends - run_starts + 1

            if run_lens.size > 0 and np.max(run_lens) > N_spike_max:
                is_front_long_straight = False

        # ---向前看滑移率---
        s_corner_min = s_fail_center - 50
        s_corner_max = s_fail_center
        mask_corner_center = (s >= s_corner_min) & (s <= s_corner_max) & (abs_kappa > thr_straight)
        mask_corner_seg = mask_corner_center[idx_center_for_seg]
        idx_corner_seg = np.where(mask_corner_seg)[0]

        SlipAngle_fl = segments["SlipAngle_fl"].to_numpy()[idx_corner_seg]
        SlipAngle_fr = segments["SlipAngle_fr"].to_numpy()[idx_corner_seg]
        SlipAngle_rl = segments["SlipAngle_rl"].to_numpy()[idx_corner_seg]
        SlipAngle_rr = segments["SlipAngle_rr"].to_numpy()[idx_corner_seg]

        sr_fl = segments["tyre_slip_ratio_fl"].to_numpy()[idx_corner_seg]
        sr_fr = segments["tyre_slip_ratio_fr"].to_numpy()[idx_corner_seg]
        sr_rl = segments["tyre_slip_ratio_rl"].to_numpy()[idx_corner_seg]
        sr_rr = segments["tyre_slip_ratio_rr"].to_numpy()[idx_corner_seg]

        # 前后轴平均侧偏角 & 滑移率
        slipF = np.mean((np.abs(SlipAngle_fl) + np.abs(SlipAngle_fr)) / 2.0)
        slipR = np.mean((np.abs(SlipAngle_rl) + np.abs(SlipAngle_rr)) / 2.0)
        srF = np.mean((np.abs(sr_fl) + np.abs(sr_fr)) / 2.0)
        srR = np.mean((np.abs(sr_rl) + np.abs(sr_rr)) / 2.0)
        
        steer = segments["steerAngle"].to_numpy()[idx_corner_seg]
        mean_steer = np.mean(np.abs(steer))

        ratio_slip_angle = slipF / max(slipR, 1e-3)
        ratio_slip_ratio = srF / max(srR, 1e-3)

        ratio_slip_angle_R = slipR / max(slipF, 1e-3)
        ratio_slip_ratio_R = srR / max(srF, 1e-3)

        understeer_flag = (ratio_slip_angle > 1.2) and (ratio_slip_ratio > 1.1) and (mean_steer > 5.0)
        oversteer_flag = (ratio_slip_angle_R > 1.2) and (ratio_slip_ratio_R > 1.1)

        if understeer_flag or oversteer_flag:
            if is_front_long_straight and is_back_long_straight:
                reason ="steer Wrong"
            elif is_back_long_straight:
                reason ="Late Braking"
            elif is_front_long_straight:
                reason ="Early Throttle"
            else:
                if understeer_flag and not oversteer_flag:#推头
                    reason ="Understeer"
                elif oversteer_flag and not understeer_flag:#摆尾
                    reason ="Oversteer"
                elif understeer_flag and oversteer_flag:
                    reason ="Understeer and Oversteer"
        else:
            reason ="steer Wrong"

        failure_info = {
            "failure_reason": reason,
            "failure_LapDist": segments["LapDist"].to_numpy()[-1],
        }
        # input("按回车继续...")
        return failure_info

class ReferenceTrack:

    file_channels_left = ["left_border_x", "left_border_y"]
    file_channels_right = ["right_border_x", "right_border_y"]
    target_speed_channel_name = "target_speed"
    channels      = ["position.x", "position.y", "lapDistance", "yaw", "curvature"]
    channels_dist = ["position.x", "position.y", "lapDistance", "yaw", "curvature"]

    def __init__(self, file_path, use_target_speed,side):
        logger.info(f"Reference Lap. Loading: {file_path}")
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.df = self.df.reset_index()
        self.use_target_speed = use_target_speed
        
        if side:
            self.file_channels=self.file_channels_left
        else:
            self.file_channels=self.file_channels_right

        try:
            self.ts = self.df[self.file_channels].values
            self.left=self.df[self.file_channels_left].values
            self.right=self.df[self.file_channels_right].values
            self.mid=(self.right+self.left)/2
        except KeyError:
            logger.error(f"Channels {self.file_channels} not found.")
            logger.error(f"Channels in racing line file: {self.df.columns}")
            raise

        if self.use_target_speed:
            self.target_speed = self.df[self.target_speed_channel_name].values.reshape(-1,1)
            print(self.target_speed)
            logger.info("Using target speed")
            self.channels.append(self.target_speed_channel_name)
            self.channels_dist.append(self.target_speed_channel_name)

        # calculate distance channels from x,y coordinates
        self.distance_ch_time = calculate_distance_from_xy(self.ts[:,0], self.ts[:,1])
        self.ts = np.concatenate([self.ts, self.distance_ch_time.reshape(-1,1)], axis=1)

        # calculate angle_y from x,y coordinates
        # calculate the yaw, fl gives a yaw wrapped to pi
        yaw = get_yaw(self.ts[:,0], self.ts[:,1])
        yaw = np.insert(yaw, 0, yaw[0]).reshape(-1,1)
        self.ts = np.concatenate([self.ts, yaw], axis=1)

        # calculate curvature
        # If the curvature is present in the racing line use it else calculate it
        if "curvature" in self.df.columns:
            logger.info("Using curvature from racing line file")
            curvatures = self.df["curvature"].values.reshape(-1,1)
        else:
            logger.info("Calculating curvature")
            curvatures = curvature_splines( self.ts[:,0],  self.ts[:,1] )
            curvatures = curvatures.reshape(-1,1)
        self.ts = np.concatenate([self.ts, curvatures], axis=1)

        if self.use_target_speed:
            self.ts = np.concatenate([self.ts, self.target_speed], axis=1)

        # interpolate to distance
        td = []
        for _, ch in enumerate(self.channels_dist):
            idx = self.channels.index(ch)
            td.append( convert_to_distance(self.distance_ch_time, self.ts[:,idx]) )
        self.td = np.array( td ).T
        self.distance_ch_dist = self.td[:,2]

        # self.df['Lapdist'] = self.ts[:,2]
        # self.df.to_csv(self.file_path, index=False)

    def get_racing_line_time(self):
        return self.ts[:, 0:2]

    def get_racing_line_dist(self):
        return self.td[:, 0:2]

    def get_channel_time(self, channel_name):
        return self.ts[:,self.channels.index(channel_name)].reshape(-1,1)

    def get_channel_dist(self, channel_name):
        return self.td[:,self.channels.index(channel_name)].reshape(-1,1)

    """
    Curvature look ahead
    """
    def distSegment2Index(self, rl_dist, l_bound, u_bound):
        return np.where((rl_dist >= l_bound) & (rl_dist <= u_bound))[0]

    def getLADVector(self, rl_dist, dist, LA_dist, vector_size, channel):
        """
            Get a vector (len vector_size) of a channel at max LA_dist

            rl_dist: time series with the distance channel interpolated and projected to distance
            dist: current distance of the car
            LA_dist: how far to look ahead [m]
            vector_size: downsample the result to this value
            channel: distance interpolated channel
            returns: vector of vector_size with the channel interpolated by distance

        """
        rl_dist = rl_dist.copy()
        patch = 0
        track_len = rl_dist[-1]

        if ((dist - track_len) > 50):
            print("## look ahead was out of range!!! Will return a Zero Vector", dist, track_len)
            assert ((dist - track_len) > 50), "distance was more than 50 meters bigger than the track len dist %f track_len %f" \
                                               % (dist, track_len)

        start = dist
        end = dist + LA_dist
        segment = self.distSegment2Index(rl_dist, start, end)

        if end > track_len:
            patch = end - track_len
            segment = np.concatenate( [segment, self.distSegment2Index(rl_dist, 0, patch)] )

        vector = channel[segment]
        vector = vector[0::len(vector) // vector_size]
        vector = vector[0:vector_size]
        return vector, segment, patch

    def get_curvature_segment(self, dist, LA_dist, vector_size):
        """
        Get single value curvature:
            dist: starting distance in the racing line
            LA_dist: lookahed starting from dist
            vector_size: downsampled signal size
        """
        curv_index = self.channels_dist.index("curvature")
        vector, segment, patch = self.getLADVector(self.distance_ch_dist, dist, LA_dist, vector_size, self.td[:, curv_index])
        return vector

    def get_target_speed_segment(self, dist, LA_dist, vector_size):
        """
        Get single value curvature:
            dist: starting distance in the racing line
            LA_dist: lookahed starting from dist
            vector_size: downsampled signal size
        """
        assert self.use_target_speed, "target speed not used"

        target_speed_index = self.channels_dist.index("target_speed")
        vector, segment, patch = self.getLADVector(self.distance_ch_dist, dist, LA_dist, vector_size, self.td[:, target_speed_index])
        return vector

    def get_target_speed_value(self, dist):
        """
        Get single value of the target speed
        """
        assert self.use_target_speed, "target speed not enabled"

        target_speed_index = self.channels_dist.index("target_speed")
        vector, segment, patch = self.getLADVector(self.distance_ch_dist, dist, 200., 1, self.td[:, target_speed_index])
        return vector[0]

    def get_yaw(self, dist):
        """
        Get single value curvature
        """
        curv_index = self.channels_dist.index("yaw")
        vector, segment, patch = self.getLADVector(self.distance_ch_dist, dist, 200., 1, self.td[:,curv_index])
        return vector[0]

    def get_curvature(self, dist):
        """
        Get single value curvature
        """
        curv_index = self.channels_dist.index("curvature")
        vector, segment, patch = self.getLADVector(self.distance_ch_dist, dist, 200., 1, self.td[:,curv_index])
        return vector

    def cropped_racing_line(self, start, segment_len, vector_len):
        pos_x_idx = self.channels_dist.index("pos_x")
        pos_y_idx = self.channels_dist.index("pos_y")
        racing_line_cropped_x, _, _ = self.getLADVector(self.distance_ch_dist, start,
                                                        segment_len, vector_len, self.td[:,pos_x_idx])
        racing_line_cropped_y, _, _ = self.getLADVector(self.distance_ch_dist, start,
                                                        segment_len, vector_len, self.td[:,pos_y_idx])
        return np.vstack([ racing_line_cropped_x, racing_line_cropped_y ]).T

    def get_track_segment(self,lapdist):
        col=self.ts[:,2]
        idx = np.where((col >= lapdist-300) & (col <= lapdist+100))[0]
        start = idx[0]
        end   = idx[-1]
        left_seg   = self.left[start:end+1, :]     # (N, 2): [x,y]
        right_seg  = self.right[start:end+1, :]    # (N, 2)
        df = pd.DataFrame({
            "left_border_x":  left_seg[:, 0],
            "left_border_y":  left_seg[:, 1],
            "right_border_x": right_seg[:, 0],
            "right_border_y": right_seg[:, 1],
            "LapDist": col[start:end+1],
        })
        return df

if __name__ == "__main__":
    barcelona_track= "F:/code/assetto_corsa_gym-main/assetto_corsa_gym/AssettoCorsaConfigs/tracks/ks_barcelona-layout_gp.csv"
    barcelona_line= "F:/code/assetto_corsa_gym-main/assetto_corsa_gym/AssettoCorsaConfigs/tracks/ks_barcelona-layout_gp-racing_line_fixed.csv"
    monza_track = "F:/code/assetto_corsa_gym-main/assetto_corsa_gym/AssettoCorsaConfigs/tracks/monza.csv"
    monza_line = "F:/code/assetto_corsa_gym-main/assetto_corsa_gym/AssettoCorsaConfigs/tracks/monza-racing_line.csv"
    redbull_track = "F:/code/assetto_corsa_gym-main/assetto_corsa_gym/AssettoCorsaConfigs/tracks/ks_red_bull_ring-layout_gp.csv"
    redbull_line = "F:/code/assetto_corsa_gym-main/assetto_corsa_gym/AssettoCorsaConfigs/tracks/ks_red_bull_ring-layout_gp-racing_line.csv"

    # track_left = ReferenceTrack(redbull_track, use_target_speed=False,side=1)
    track_right = ReferenceTrack(redbull_track, use_target_speed=False,side=0)
    
    # ref_lap = ReferenceLap(redbull_line, False)
    # print(ref_lap.segments)
    # print(ref_lap.type)
