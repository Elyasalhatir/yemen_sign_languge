
import MediapipeAvatarController from "./MediapipeAvatarController.js";
import MediapipePoseCalculator from "./MediapipePoseCalculator.js";

class MediapipeAvatarManager {
    constructor() {
        this.poseCalculator = new MediapipePoseCalculator();
        this.avatarController = new MediapipeAvatarController();
        this.onFrameProcessed = null; // Callback for recorder
        this.handMode = 'both'; // 'both', 'left', 'right'
    }

    setHandMode(mode) {
        this.handMode = mode;
        console.log("Hand Mode set to:", mode);
    }

    update(results) {
        const calculatedResult = this.poseCalculator.update(results);

        // --- Filter based on Hand Mode ---
        // When one hand is disabled, put it in Rest Pose position
        // Rest Pose: Arms straight down beside the body (180 degrees / PI rotation)
        // Quaternion for 90° rotation around Z-axis: [0, 0, 0.707, 0.707] (arms down)
        // For fully down position (arms beside thighs):
        // Left arm: rotate 90° around Z = [0, 0, -0.707, 0.707]
        // Right arm: rotate -90° around Z = [0, 0, 0.707, 0.707]

        // A-Pose (Arms at roughly 45-60 degrees down)
        // Rotation around Z axis.
        // Left Arm: approx +60 degrees (rotated down from T-pose) -> Z = sin(60/2) = 0.5
        // We will use 0.0, 0.0, 0.479, 0.877 (approx 57 degrees) 

        // Let's use clean 60 degrees (PI/3): sin(PI/6)=0.5, cos(PI/6)=0.866
        // Left Arm Z rotation:
        const leftArmRestPose = [
            0,
            0,
            0.5,
            0.866
        ];   // Left arm A-Pose

        // Right Arm Z rotation (negative):
        const rightArmRestPose = [
            0,
            0,
            -0.5,
            0.866
        ];   // Right arm A-Pose
        const foreArmStraight = [0, 0, 0, 1];            // Forearm straight (identity)

        // Handle untracked hands (when landmarks not detected)
        // Put the untracked arm in rest pose by the thigh
        if (this.handMode === 'both') {
            // If left hand not tracked, put left arm in rest pose
            if (!results.leftHandLandmarks && calculatedResult.poseQuatArr) {
                calculatedResult.poseQuatArr[12] = leftArmRestPose; // LeftArm
                calculatedResult.poseQuatArr[13] = foreArmStraight;  // LeftForeArm
                calculatedResult.leftHandQuatArr = new Array(21).fill(foreArmStraight);
            }
            // If right hand not tracked, put right arm in rest pose
            if (!results.rightHandLandmarks && calculatedResult.poseQuatArr) {
                calculatedResult.poseQuatArr[8] = rightArmRestPose; // RightArm
                calculatedResult.poseQuatArr[9] = foreArmStraight;  // RightForeArm
                calculatedResult.rightHandQuatArr = new Array(21).fill(foreArmStraight);
            }
        } else if (this.handMode === 'left') {
            // Disable Right Hand -> Put in Rest Pose
            if (calculatedResult.poseQuatArr) {
                calculatedResult.poseQuatArr[8] = rightArmRestPose; // RightArm - Rest
                calculatedResult.poseQuatArr[9] = foreArmStraight;  // RightForeArm - Straight
            }

            // Force Right Hand to Open/Relaxed position
            calculatedResult.rightHandQuatArr = new Array(21).fill(foreArmStraight);

        } else if (this.handMode === 'right') {
            // Disable Left Hand -> Put in Rest Pose
            if (calculatedResult.poseQuatArr) {
                calculatedResult.poseQuatArr[12] = leftArmRestPose; // LeftArm - Rest
                calculatedResult.poseQuatArr[13] = foreArmStraight; // LeftForeArm - Straight
            }

            // Force Left Hand to Open/Relaxed position
            calculatedResult.leftHandQuatArr = new Array(21).fill(foreArmStraight);
        }

        this.avatarController.update(calculatedResult);

        if (this.onFrameProcessed) {
            this.onFrameProcessed(calculatedResult);
        }
    }

    bindAvatar(avatar, type) {
        this.avatarController.bindAvatar(avatar, type);
    }

    setUseHand(useHand) {
        this.poseCalculator.setUseHand(useHand);
        this.avatarController.setUseHand(useHand);
    }

    setUseFace(useFace) {
        this.avatarController.setUseFace(useFace);
    }

    setSlerpRatio(ratio) {
        this.avatarController.setSlerpRatio(ratio);
    }

    initKalmanFilter() {
        this.poseCalculator.initKalmanFilter();
        this.avatarController.initKalmanFilter();
    }

    setUseKalmanFilter(useKalmanFilter) {
        this.poseCalculator.setUseKalmanFilter(useKalmanFilter);
        this.avatarController.setUseKalmanFilter(useKalmanFilter);
    }
}

export default MediapipeAvatarManager;
