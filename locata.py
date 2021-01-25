# LOCATA dataset loader
from torch.utils.data import Dataset
import webrtcvad
import soundfile
import pandas
from collections import namedtuple
import numpy as np
import os
# Named tuple with the characteristics of a microphone array and definitions of the LOCATA arrays:
ArraySetup = namedtuple('ArraySetup', 'arrayType, orV, mic_pos, mic_orV, mic_pattern')
benchmark2_array_setup = ArraySetup(arrayType='3D', 
    orV = np.array([0.0, 1.0, 0.0]),
    mic_pos = np.array(((-0.028,  0.030, -0.040),
                        ( 0.006,  0.057,  0.000),
                        ( 0.022,  0.022, -0.046),
                        (-0.055, -0.024, -0.025),
                        (-0.031,  0.023,  0.042),
                        (-0.032,  0.011,  0.046),
                        (-0.025, -0.003,  0.051),
                        (-0.036, -0.027,  0.038),
                        (-0.035, -0.043,  0.025),
                        ( 0.029, -0.048, -0.012),
                        ( 0.034, -0.030,  0.037),
                        ( 0.035,  0.025,  0.039))), 
    mic_orV = np.array(((-0.028,  0.030, -0.040),
                        ( 0.006,  0.057,  0.000),
                        ( 0.022,  0.022, -0.046),
                        (-0.055, -0.024, -0.025),
                        (-0.031,  0.023,  0.042),
                        (-0.032,  0.011,  0.046),
                        (-0.025, -0.003,  0.051),
                        (-0.036, -0.027,  0.038),
                        (-0.035, -0.043,  0.025),
                        ( 0.029, -0.048, -0.012),
                        ( 0.034, -0.030,  0.037),
                        ( 0.035,  0.025,  0.039))),
    mic_pattern = 'omni'
)
def cart2sph(cart):
    xy2 = cart[:,0]**2 + cart[:,1]**2
    sph = np.zeros_like(cart)
    sph[:,0] = np.sqrt(xy2 + cart[:,2]**2)
    sph[:,1] = np.arctan2(np.sqrt(xy2), cart[:,2]) # Elevation angle defined from Z-axis down
    sph[:,2] = np.arctan2(cart[:,1], cart[:,0])
    return sph


class AcousticScene:
    """ Acoustic scene class.
    It contains everything needed to simulate a moving sound source moving recorded
    with a microphone array in a reverberant room.
    It can also store the results from the DOA estimation.
    """
    def __init__(self, room_sz, T60, beta, SNR, array_setup, mic_pos, source_signal, fs, traj_pts, timestamps,
                 trajectory, t, DOA):
        self.room_sz = room_sz                # Room size
        self.T60 = T60                        # Reverberation time of the simulated room
        self.beta = beta                    # Reflection coefficients of the walls of the room (make sure it corresponds with T60)
        self.SNR = SNR                        # Signal to (omnidirectional) Noise Ration to simulate
        self.array_setup = array_setup        # Named tuple with the characteristics of the array
        self.mic_pos = mic_pos                # Position of the center of the array
        self.source_signal = source_signal  # Source signal
        self.fs = fs                        # Sampling frequency of the source signal and the simulations
        self.traj_pts = traj_pts             # Trajectory points to simulate
        self.timestamps = timestamps        # Time of each simulation (it does not need to correspond with the DOA estimations)
        self.trajectory = trajectory        # Continuous trajectory
        self.t = t                            # Continuous time
        self.DOA = DOA                         # Continuous DOA

    def simulate(self):
        """ Get the array recording using gpuRIR to perform the acoustic simulations.
        """
        if self.T60 == 0:
            Tdiff = 0.1
            Tmax = 0.1
            nb_img = [1,1,1]
        else:
            Tdiff = gpuRIR.att2t_SabineEstimator(12, self.T60) # Use ISM until the RIRs decay 12dB
            Tmax = gpuRIR.att2t_SabineEstimator(40, self.T60)  # Use diffuse model until the RIRs decay 40dB
            if self.T60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
            nb_img = gpuRIR.t2n( Tdiff, self.room_sz )

        nb_mics  = len(self.mic_pos)
        nb_traj_pts = len(self.traj_pts)
        nb_gpu_calls = min(int(np.ceil( self.fs * Tdiff * nb_mics * nb_traj_pts * np.prod(nb_img) / 1e9 )), nb_traj_pts)
        traj_pts_batch = np.ceil( nb_traj_pts / nb_gpu_calls * np.arange(0, nb_gpu_calls+1) ).astype(int)

        RIRs_list = [ gpuRIR.simulateRIR(self.room_sz, self.beta,
                         self.traj_pts[traj_pts_batch[0]:traj_pts_batch[1],:], self.mic_pos,
                         nb_img, Tmax, self.fs, Tdiff=Tdiff,
                         orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern) ]
        for i in range(1,nb_gpu_calls):
            RIRs_list += [    gpuRIR.simulateRIR(self.room_sz, self.beta,
                             self.traj_pts[traj_pts_batch[i]:traj_pts_batch[i+1],:], self.mic_pos,
                             nb_img, Tmax, self.fs, Tdiff=Tdiff,
                             orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern) ]
        RIRs = np.concatenate(RIRs_list, axis=0)
        mic_signals = gpuRIR.simulateTrajectory(self.source_signal, RIRs, timestamps=self.timestamps, fs=self.fs)
        mic_signals = mic_signals[0:len(self.t),:]

        # Omnidirectional noise
        dp_RIRs = gpuRIR.simulateRIR(self.room_sz, self.beta, self.traj_pts, self.mic_pos, [1,1,1], 0.1, self.fs,
                                    orV_rcv=self.array_setup.mic_orV, mic_pattern=self.array_setup.mic_pattern)
        dp_signals = gpuRIR.simulateTrajectory(self.source_signal, dp_RIRs, timestamps=self.timestamps, fs=self.fs)
        ac_pow = np.mean([acoustic_power(dp_signals[:,i]) for i in range(dp_signals.shape[1])])
        noise = np.sqrt(ac_pow/10**(self.SNR/10)) * np.random.standard_normal(mic_signals.shape)
        mic_signals += noise

        # Apply the propagation delay to the VAD information if it exists
        if hasattr(self, 'source_vad'):
            vad = gpuRIR.simulateTrajectory(self.source_vad, dp_RIRs, timestamps=self.timestamps, fs=self.fs)
            self.vad = vad[0:len(self.t),:].mean(axis=1) > vad[0:len(self.t),:].max()*1e-3

        return mic_signals

    def get_rmsae(self, exclude_silences=False):
        """ Returns the Root Mean Square Angular Error (degrees) of the DOA estimation.
        The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
        """
        if not exclude_silences:
            return rms_angular_error_deg(torch.from_numpy(self.DOAw_pred).double(),
                                         torch.from_numpy(self.DOAw).double() )
        else:
            silences = self.vad.mean(axis=1) < 2/3
            DOAw_pred = torch.from_numpy(self.DOAw_pred[np.invert(silences), :]).double()
            self.DOAw_pred[silences, :] = np.NaN
            DOAw = torch.from_numpy(self.DOAw[np.invert(silences), :]).double()
            return rms_angular_error_deg(DOAw_pred, DOAw)

    def findMapMaximum(self, exclude_silences=False):
        """ Generates the field DOAw_est_max with the DOA estimation using the SRP-PHAT maximums
        and returns its RMSAE (in degrees) if the field DOAw exists with the DOA groundtruth.
        The scene need to have the field maps with the SRP-PHAT map of each window.
        You can choose whether to include the silent frames into the RMSAE computation or not.
        """
        max_flat_idx = self.maps.reshape((self.maps.shape[0], -1)).argmax(1)
        theta_max_idx, phi_max_idx = np.unravel_index(max_flat_idx, self.maps.shape[1:])

        # Index to angle (radians)
        if self.array_setup.arrayType == 'planar':
            theta = np.linspace(0, np.pi/2, self.maps.shape[1])
        else:
            theta= np.linspace(0, np.pi, self.maps.shape[1])
        phi = np.linspace(-np.pi, np.pi, self.maps.shape[2]+1)
        phi = phi[:-1]
        DOAw_srpMax = np.stack((theta[theta_max_idx], phi[phi_max_idx]), axis=-1)
        self.DOAw_srpMax = DOAw_srpMax

        if not exclude_silences:
            if hasattr(self, 'DOAw'):
                return rms_angular_error_deg(torch.from_numpy(self.DOAw_srpMax),
                                                        torch.from_numpy(self.DOAw))
        else:
            silences = self.vad.mean(axis=1) < 2/3
            self.DOAw_srpMax[silences] = np.NaN
            if hasattr(self, 'DOAw'):
                return rms_angular_error_deg(torch.from_numpy(DOAw_srpMax[np.invert(silences), :]),
                                                         torch.from_numpy(self.DOAw[np.invert(silences), :]) )

    def plotScene(self, view='3D'):
        """ Plots the source trajectory and the microphones within the room
        """
        assert view in ['3D', 'XYZ', 'XY', 'XZ', 'YZ']

        fig = plt.figure()

        if view == '3D' or view == 'XYZ':
            ax = Axes3D(fig)
            ax.set_xlim3d(0, self.room_sz[0])
            ax.set_ylim3d(0, self.room_sz[1])
            ax.set_zlim3d(0, self.room_sz[2])

            ax.scatter(self.traj_pts[:,0], self.traj_pts[:,1], self.traj_pts[:,2])
            ax.scatter(self.mic_pos[:,0], self.mic_pos[:,1], self.mic_pos[:,2])
            ax.text(self.traj_pts[0,0], self.traj_pts[0,1], self.traj_pts[0,2], 'start')

            ax.set_title('$T_{60}$' + ' = {:.3f}s, SNR = {:.1f}dB'.format(self.T60, self.SNR))
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')

        else:
            ax = fig.add_subplot(111)
            plt.gca().set_aspect('equal', adjustable='box')

            if view == 'XY':
                ax.set_xlim(0, self.room_sz[0])
                ax.set_ylim(0, self.room_sz[1])
                ax.scatter(self.traj_pts[:,0], self.traj_pts[:,1])
                ax.scatter(self.mic_pos[:,0], self.mic_pos[:,1])
                ax.text(self.traj_pts[0,0], self.traj_pts[0,1], 'start')
                ax.legend(['Source trajectory', 'Microphone array'])
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
            elif view == 'XZ':
                ax.set_xlim(0, self.room_sz[0])
                ax.set_ylim(0, self.room_sz[2])
                ax.scatter(self.traj_pts[:,0], self.traj_pts[:,2])
                ax.scatter(self.mic_pos[:,0], self.mic_pos[:,2])
                ax.text(self.traj_pts[0,0], self.traj_pts[0,2], 'start')
                ax.legend(['Source trajectory', 'Microphone array'])
                ax.set_xlabel('x [m]')
                ax.set_ylabel('z [m]')
            elif view == 'YZ':
                ax.set_xlim(0, self.room_sz[1])
                ax.set_ylim(0, self.room_sz[2])
                ax.scatter(self.traj_pts[:,1], self.traj_pts[:,2])
                ax.scatter(self.mic_pos[:,1], self.mic_pos[:,2])
                ax.text(self.traj_pts[0,1], self.traj_pts[0,2], 'start')
                ax.legend(['Source trajectory', 'Microphone array'])
                ax.set_xlabel('y [m]')
                ax.set_ylabel('z [m]')

        plt.show()

    def plotDOA(self):
        """ Plots the groundtruth DOA
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.t, self.DOA * 180/np.pi)

        ax.legend(['Elevation', 'Azimuth'])
        ax.set_xlabel('time [s]')
        ax.set_ylabel('DOA [º]')

        plt.show()

    def plotEstimation(self, legned_loc='best'):
        """ Plots the DOA groundtruth and its estimation.
        The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
        If the scene has the field DOAw_srpMax with the SRP-PHAT estimation, it also plots it.
        """
        fig = plt.figure()
        gs = fig.add_gridspec(7, 1)
        ax = fig.add_subplot(gs[0,0])
        ax.plot(self.t, self.source_signal)
        plt.xlim(self.tw[0], self.tw[-1])
        plt.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)

        ax = fig.add_subplot(gs[1:,0])
        ax.plot(self.tw, self.DOAw * 180/np.pi)
        plt.gca().set_prop_cycle(None)
        ax.plot(self.tw, self.DOAw_pred * 180/np.pi, '--')
        if hasattr(self, 'DOAw_srpMax'):
            plt.gca().set_prop_cycle(None)
            ax.plot(self.tw, self.DOAw_srpMax * 180 / np.pi, 'x', markersize=4)

        plt.legend(['Elevation', 'Azimuth'], loc=legned_loc)
        plt.xlabel('time [s]')
        plt.ylabel('DOA [º]')

        silences = self.vad.mean(axis=1) < 2/3
        silences_idx = silences.nonzero()[0]
        start, end = [], []
        for i in silences_idx:
            if not i - 1 in silences_idx:
                start.append(i)
            if not i + 1 in silences_idx:
                end.append(i)
        for s, e in zip(start, end):
            plt.axvspan((s-0.5)*self.tw[1], (e+0.5)*self.tw[1], facecolor='0.5', alpha=0.5)

        plt.xlim(self.tw[0], self.tw[-1])
        plt.show()


    def plotMap(self, w_idx):
        """ Plots the SRP-PHAT map of the window w_idx.
        If the scene has the fields DOAw, DOAw_pred, DOAw_srpMax it also plot them.
        """
        maps = np.concatenate((self.maps, self.maps[..., 0, np.newaxis]), axis=-1)

        thetaMax = np.pi / 2 if self.array_setup.arrayType == 'planar' else np.pi
        theta = np.linspace(0, thetaMax, maps.shape[-2])
        phi = np.linspace(-np.pi, np.pi, maps.shape[-1])

        map = maps[w_idx, ...]
        DOA = self.DOAw[w_idx, ...] if hasattr(self, 'DOAw') else None
        DOA_pred = self.DOAw_pred[w_idx, ...] if hasattr(self, 'DOAw_pred') else None
        DOA_srpMax = self.DOAw_srpMax[w_idx, ...] if hasattr(self, 'DOAw_srpMax') else None

        plot_srp_map(theta, phi, map, DOA, DOA_pred, DOA_srpMax)

    def animateScene(self, fps=10, file_name=None):
        """ Creates an animation with the SRP-PHAT maps of each window.
        The scene need to have the field maps with the SRP-PHAT map of each window.
        If the scene has the fields DOAw, DOAw_pred, DOAw_srpMax it also includes them.
        """
        maps = np.concatenate((self.maps, self.maps[..., 0, np.newaxis]), axis=-1)
        thetaMax = np.pi/2 if self.array_setup=='planar' else np.pi
        theta = np.linspace(0, thetaMax, maps.shape[-2])
        phi = np.linspace(-np.pi, np.pi, maps.shape[-1])

        DOAw = self.DOAw if hasattr(self, 'DOAw') else None
        DOAw_pred = self.DOAw_pred if hasattr(self, 'DOAw_pred') else None
        DOAw_srpMax = self.DOAw_srpMax if hasattr(self, 'DOAw_srpMax') else None

        animate_trajectory(theta, phi, maps, fps, DOAw, DOAw_pred, DOAw_srpMax, file_name)


class  LocataDataset(Dataset):
    """ Dataset with the LOCATA dataset recordings and its corresponding Acoustic Scenes.
    When you access to an element you get both the simulated signals in the microphones and the AcousticScene object.
    """
    def __init__(self, path, array, fs, tasks=(1,3,5), recording=None, dev=False, transforms = None):
        """
        path: path to the root of the LOCATA dataset in your file system
        array: string with the desired array ('dummy', 'eigenmike', 'benchmark2' or 'dicit')
        fs: sampling frequency (you can use it to downsample the LOCATA recordings)
        tasks: LOCATA tasks to include in the dataset (only one-source tasks are supported)
        recording: recordings that you want to include in the dataset (only supported if you selected only one task)
        dev: True if the groundtruth source positions are available
        transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene
        """
        assert array in ('dummy', 'eigenmike', 'benchmark2', 'dicit'), 'Invalid array.'
        assert recording is None or len(tasks) == 1, 'Specific recordings can only be selected for dataset with only one task'
        for task in tasks: assert task in (1,3,5), 'Invalid task ' + str(task) + '.'

        self.path = path
        self.dev = dev
        self.array = array
        self.tasks = tasks
        self.transforms = transforms
        self.fs = fs

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)

        if array == 'dummy':
            self.array_setup = dummy_array_setup
        elif array == 'eigenmike':
            self.array_setup = eigenmike_array_setup
        elif array == 'benchmark2':
            self.array_setup = benchmark2_array_setup
        elif array == 'dicit':
            self.array_setup = dicit_array_setup

        self.directories = []
        for task in tasks:
            task_path = os.path.join(path, 'task' + str(task))
            for recording in os.listdir( task_path ):
                arrays = os.listdir( os.path.join(task_path, recording) )
                if array in arrays:
                    self.directories.append( os.path.join(task_path, recording, array) )
        self.directories.sort()

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, idx):
        directory = self.directories[idx]
        mic_signals, fs = soundfile.read( os.path.join(directory, 'audio_array_' + self.array + '.wav') )
        if fs > self.fs:
            mic_signals = scipy.signal.decimate(mic_signals, int(fs/self.fs), axis=0)
            new_fs = fs / int(fs/self.fs)
            if new_fs != self.fs: warnings.warn('The actual fs is {}Hz'.format(new_fs))
            self.fs = new_fs
        elif fs < self.fs:
            raise Exception('The sampling rate of the file ({}Hz) was lower than self.fs ({}Hz'.format(fs, self.fs))

        # Remove initial silence
        start = np.argmax(mic_signals[:,0] > mic_signals[:,0].max()*0.15)
        mic_signals = mic_signals[start:,:]
        t = (np.arange(len(mic_signals)) + start)/self.fs

        df = pandas.read_csv( os.path.join(directory, 'position_array_' + self.array + '.txt'), sep='\t' )
        array_pos = np.stack((df['x'].values, df['y'].values,df['z'].values), axis=-1)
        array_ref_vec = np.stack((df['ref_vec_x'].values, df['ref_vec_y'].values,df['ref_vec_z'].values), axis=-1)
        array_rotation = np.zeros((array_pos.shape[0],3,3))
        for i in range(3):
            for j in range(3):
                array_rotation[:,i,j] = df['rotation_' + str(i+1) + str(j+1)]

        df = pandas.read_csv( os.path.join(directory, 'required_time.txt'), sep='\t' )
        required_time = df['hour'].values*3600+df['minute'].values*60+df['second'].values
        timestamps = required_time - required_time[0]

        if self.dev:
            sources_pos = []
            trajectories = []
            for file in os.listdir( directory ):
                if file.startswith('audio_source') and file.endswith('.wav'):
                    source_signal, fs_src = soundfile.read(os.path.join(directory, file))
                    if fs > self.fs:
                        source_signal = scipy.signal.decimate(source_signal, int(fs_src / self.fs), axis=0)
                    source_signal = source_signal[start:start+len(t)]
                if file.startswith('position_source'):
                    df = pandas.read_csv( os.path.join(directory, file), sep='\t' )
                    source_pos = np.stack((df['x'].values, df['y'].values,df['z'].values), axis=-1)
                    sources_pos.append( source_pos )
                    trajectories.append( np.array([np.interp(t, timestamps, source_pos[:,i]) for i in range(3)]).transpose() )
            sources_pos = np.stack(sources_pos)
            trajectories = np.stack(trajectories)

            DOA_pts = np.zeros(sources_pos.shape[0:2] + (2,))
            DOA = np.zeros(trajectories.shape[0:2] + (2,))
            for s in range(sources_pos.shape[0]):
                source_pos_local = np.matmul( np.expand_dims(sources_pos[s,...] - array_pos, axis=1), array_rotation ).squeeze() # np.matmul( array_rotation, np.expand_dims(sources_pos[s,...] - array_pos, axis=-1) ).squeeze()
                DOA_pts[s,...] = cart2sph(source_pos_local) [:,1:3]
                DOA[s,...] = np.array([np.interp(t, timestamps, DOA_pts[s,:,i]) for i in range(2)]).transpose()
            DOA[DOA[...,1]<-np.pi, 1] += 2*np.pi
        else:
            sources_pos = None
            DOA = None
            source_signal = np.NaN * np.ones((len(mic_signals),1))

        acoustic_scene = AcousticScene(
            room_sz = np.NaN * np.ones((3,1)),
            T60 = np.NaN,
            beta = np.NaN * np.ones((6,1)),
            SNR = np.NaN,
            array_setup = self.array_setup,
            mic_pos = np.matmul( array_rotation[0,...], np.expand_dims(self.array_setup.mic_pos, axis=-1) ).squeeze() + array_pos[0,:], # self.array_setup.mic_pos + array_pos[0,:], # Not valid for moving arrays
            source_signal = source_signal,
            fs = self.fs,
            t = t - start/self.fs,
            traj_pts = sources_pos[0,...],
            timestamps = timestamps - start/self.fs,
            trajectory = trajectories[0,...],
            DOA = DOA[0,...]
        )

        vad = np.zeros_like(source_signal)
        vad_frame_len = int(10e-3 * self.fs)
        n_vad_frames = len(source_signal) // vad_frame_len
        for frame_idx in range(n_vad_frames):
            frame = source_signal[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
            frame_bytes = (frame * 32767).astype('int16').tobytes()
            vad[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = self.vad.is_speech(frame_bytes, int(self.fs))
        acoustic_scene.vad = vad

        mic_signals.transpose()
        if self.transforms is not None:
            for t in self.transforms:
                mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)

        return mic_signals, acoustic_scene

 

