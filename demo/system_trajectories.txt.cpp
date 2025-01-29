vector<Eigen::Matrix4f> System::GetCameraTrajectory() 
{
    // Check if system is ready
    if (mpTracker->mState == Tracking::NOT_INITIALIZED || 
        mpTracker->mState == Tracking::NO_IMAGES_YET)
    {
        return vector<Eigen::Matrix4f>();
    }

    // 1. Acquire lock to ensure thread safety during trajectory computation
    unique_lock<mutex> lock(mpAtlas->mutexMapUpdate);

    // 2. Get keyframes and check validity
    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    if (vpKFs.empty()) {
        return vector<Eigen::Matrix4f>();
    }
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // 3. Get the first valid keyframe to use as reference
    KeyFrame* pFirstKF = nullptr;
    for (auto* pKF : vpKFs) {
        if (pKF && !pKF->isBad()) {
            pFirstKF = pKF;
            break;
        }
    }
    if (!pFirstKF) {
        return vector<Eigen::Matrix4f>();
    }

    // 4. Transform all keyframes so that the first keyframe is at the origin
    Sophus::SE3f Two = pFirstKF->GetPoseInverse();
    vector<Eigen::Matrix4f> trajectory;

    // 5. Safely access tracker data
    if (!mpTracker || mpTracker->mlpReferences.empty()) {
        return vector<Eigen::Matrix4f>();
    }

    // 6. Iterate through frame poses with bounds checking
    auto lRit = mpTracker->mlpReferences.begin();
    auto lT = mpTracker->mlFrameTimes.begin();
    auto lbL = mpTracker->mlbLost.begin();
    
    for (auto lit = mpTracker->mlRelativeFramePoses.begin();
         lit != mpTracker->mlRelativeFramePoses.end() && 
         lRit != mpTracker->mlpReferences.end() && 
         lT != mpTracker->mlFrameTimes.end() && 
         lbL != mpTracker->mlbLost.end();
         ++lit, ++lRit, ++lT, ++lbL)
    {
        // Skip frames marked as lost
        if (*lbL) continue;

        KeyFrame* pKF = *lRit;
        if (!pKF) continue;

        Sophus::SE3f Trw;

        // Handle bad keyframes by traversing up the spanning tree
        while (pKF && pKF->isBad()) {
            KeyFrame* pParent = pKF->GetParent();
            if (!pParent) break;
            Trw = Trw * pKF->mTcp;
            pKF = pParent;
        }

        // Skip if no valid keyframe was found in the parent chain
        if (!pKF || pKF->isBad()) continue;

        // Compute the final transformation
        Trw = Trw * pKF->GetPose() * Two;
        Sophus::SE3f Tcw = (*lit) * Trw;
        Sophus::SE3f Twc = Tcw.inverse();
        trajectory.push_back(Twc.matrix());
    }

    return trajectory;
}
