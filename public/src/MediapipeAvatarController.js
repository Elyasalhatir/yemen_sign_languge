import { BoneManager } from './elementManagers.js';
import {
    MEDIAPIPE_VIRTUAL_BONES_CONFIG,
    MEDIAPIPE_LEFT_HAND_BONES_CONFIG,
    MEDIAPIPE_RIGHT_HAND_BONES_CONFIG
} from './MediapipeConfig.js';

class MediapipeAvatarController {
    constructor() {
        let leftHandBoneManager = this.leftHandBoneManager = new BoneManager();
        leftHandBoneManager.configure(MEDIAPIPE_LEFT_HAND_BONES_CONFIG);

        let rightHandBoneManager = this.rightHandBoneManager = new BoneManager();
        rightHandBoneManager.configure(MEDIAPIPE_RIGHT_HAND_BONES_CONFIG);

        let poseBoneManager = this.poseBoneManager = new BoneManager();
        poseBoneManager.configure(MEDIAPIPE_VIRTUAL_BONES_CONFIG);

        this.useHand = true;
        this.useFace = true; // Default to true

        this.setUseFace = (useFace) => {
            this.useFace = useFace;
        };

        this.initKalmanFilter = () => {
            this.poseBoneManager.initKalmanFilter();
            this.leftHandBoneManager.initKalmanFilter();
            this.rightHandBoneManager.initKalmanFilter();
        };

        this.setUseKalmanFilter = (useKalmanFilter) => {
            this.poseBoneManager.setUseKalmanFilter(useKalmanFilter);
            this.leftHandBoneManager.setUseKalmanFilter(useKalmanFilter);
            this.rightHandBoneManager.setUseKalmanFilter(useKalmanFilter);
            this.initKalmanFilter();
        };

        this.initialLegRotations = {};

        this.bindAvatar = (avatar, type, coordinateSystem) => {
            poseBoneManager.bindAvatar(avatar, type, coordinateSystem);
            leftHandBoneManager.bindAvatar(avatar, type, coordinateSystem);
            rightHandBoneManager.bindAvatar(avatar, type, coordinateSystem);

            // Capture initial leg rotations
            const legIndices = [1, 2, 4, 5];
            legIndices.forEach(index => {
                const bone = poseBoneManager.get(index);
                if (bone && poseBoneManager.avatarBones && poseBoneManager.avatarBones[index]) { // Check if avatarBone is bound
                    // Note: BoneManager stores avatarBones in an array, but 'bone' object doesn't directly have reference to the THREE.Bone easily accessible without looking at BoneManager's internal array or using getAvatarBone.
                    // However, BoneManager.getAvatarBone uses virtualBoneName.
                    // Let's use poseBoneManager.avatarBones[index] directly if accessible, or we need to access the THREE.Bone from the avatar.

                    // Actually, poseBoneManager.avatarBones is populated in bindAvatar.
                    const avatarBone = poseBoneManager.avatarBones[index];
                    if (avatarBone) {
                        this.initialLegRotations[index] = avatarBone.quaternion.clone();
                    }
                }
            });
        };

        this.setUseHand = (useHand) => {
            this.useHand = useHand;
        };

        this.setSlerpRatio = (slerpRatio) => {
            this.poseBoneManager.setSlerpRatio(slerpRatio);
            this.leftHandBoneManager.setSlerpRatio(slerpRatio);
            this.rightHandBoneManager.setSlerpRatio(slerpRatio);
        }

        this.update = (data) => {
            const poseQuatArr = data['poseQuatArr'];
            const rootPos = data['rootPos'];
            const leftHandQuatArr = data['leftHandQuatArr'];
            const rightHandQuatArr = data['rightHandQuatArr'];

            if (poseQuatArr) {
                this.updatePose(poseQuatArr, 'array');
            }
            if (rootPos) {
                this.updateRootPos(rootPos);
            }
            if (leftHandQuatArr) {
                this.updateHand(leftHandQuatArr, 'array', true);
            }
            if (rightHandQuatArr) {
                this.updateHand(rightHandQuatArr, 'array', false);
            }
        }

        this.updatePose = (poseData, dtype) => {
            // Strict Filter: Only allow Arms (8, 12), ForeArms (9, 13)
            // Conditionally allow Neck (10) and Head (11) if useFace is true
            // Indices based on MEDIAPIPE_VIRTUAL_BONES_CONFIG in MediapipeConfig.js

            let allowedIndices = [8, 9, 12, 13];
            if (this.useFace) {
                allowedIndices.push(10, 11);
            }

            // Create a filtered array where only allowed indices have data
            // We assume poseData is an array of quaternions/arrays matching the config size
            // If dtype is 'array', poseData is likely an array of arrays or objects.

            // We will manually set the bones we care about.
            // This bypasses the bulk setFromArray for the whole body.

            if (dtype === 'array') {
                allowedIndices.forEach(index => {
                    const boneData = poseData[index];
                    if (boneData) {
                        const bone = this.poseBoneManager.get(index);
                        if (bone) {
                            bone.setFromArray(boneData);
                        }
                    }
                });

                // Force Legs and Feet to Fixed Pose (Rotated 180 + Upwards)
                // Indices: 1 (LeftUpLeg), 2 (LeftLeg), 3 (LeftFoot), 4 (RightUpLeg), 5 (RightLeg), 6 (RightFoot)
                const legIndices = [1, 2, 3, 4, 5, 6];
                // Rotate 180 degrees around Z axis
                // Quaternion: x=0, y=0, z=1, w=0
                const zFlipQuat = [0, 0, 1, 0];

                legIndices.forEach(index => {
                    const bone = this.poseBoneManager.get(index);
                    if (bone) {
                        bone.setFromArray(zFlipQuat);
                    }
                });
            } else {
                // Fallback for other types if necessary, or just use full setFromArray if we trust it
                // But for strict locking, we stick to manual update.
                // If we used setFromArray(poseData), it would update hips, spine, head, etc.
            }
            // Legs (and other non-arm bones) are intentionally not updated here, ensuring they remain in their default pose.
            this.poseBoneManager.updateAvatar();
        };

        this.updateHand = (handData, dtype, isLeft) => {
            if (!this.useHand) return;
            let handBoneManager = isLeft ? this.leftHandBoneManager : this.rightHandBoneManager;
            handBoneManager.setFromArray(handData, dtype);
            handBoneManager.updateAvatar();
        };

        this.updateRootPos = (rootPos) => {
            // DISABLED: Lock avatar position completely
            // const rootBone = this.poseBoneManager.getAvatarRoot();
            // rootBone.position.fromArray(rootPos);
        }

    }
}

export default MediapipeAvatarController;