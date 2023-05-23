def spatial_distance(c1, c2):
    d = (c1[0] - c2[0])**2 +(c1[1] - c2[1])**2 
    return d**0.5

def get_features_2(handR,handL,width,height):
    
    thumb_cmc_1 = (int(handR.landmark[1].x *width),int(handR.landmark[1].y *height))
    thumb_mcp_1 = (int(handR.landmark[2].x *width),int(handR.landmark[2].y *height))
    thumb_ip_1 = (int(handR.landmark[3].x *width),int(handR.landmark[3].y *height))
    thumb_tip_1 = (int(handR.landmark[4].x *width),int(handR.landmark[4].y *height))
    index_finger_mcp_1 = (int(handR.landmark[5].x *width),int(handR.landmark[5].y *height))
    index_finger_pip_1 = (int(handR.landmark[6].x *width),int(handR.landmark[6].y *height))
    index_finger_dip_1 = (int(handR.landmark[7].x *width),int(handR.landmark[7].y *height))
    index_finger_tip_1 = (int(handR.landmark[8].x *width),int(handR.landmark[8].y *height))
    middle_finger_mcp_1 = (int(handR.landmark[9].x *width),int(handR.landmark[9].y *height))
    middle_finger_pip_1 = (int(handR.landmark[10].x *width),int(handR.landmark[10].y *height))
    middle_finger_dip_1 = (int(handR.landmark[11].x *width),int(handR.landmark[11].y *height))
    middle_finger_tip_1 = (int(handR.landmark[12].x *width),int(handR.landmark[12].y *height))
    ring_finger_mcp_1 = (int(handR.landmark[13].x *width),int(handR.landmark[13].y *height))
    ring_finger_pip_1 = (int(handR.landmark[14].x *width),int(handR.landmark[14].y *height))
    ring_finger_dip_1 = (int(handR.landmark[15].x *width),int(handR.landmark[15].y *height))
    ring_finger_tip_1 = (int(handR.landmark[16].x *width),int(handR.landmark[16].y *height))
    pinky_finger_mcp_1 = (int(handR.landmark[17].x *width),int(handR.landmark[17].y *height))
    pinky_finger_pip_1 = (int(handR.landmark[18].x *width),int(handR.landmark[18].y *height))
    pinky_finger_dip_1 = (int(handR.landmark[19].x *width),int(handR.landmark[19].y *height))
    pinky_finger_tip_1 = (int(handR.landmark[20].x *width),int(handR.landmark[20].y *height))

    thumb_cmc_2 = (int(handL.landmark[1].x *width),int(handL.landmark[1].y *height))
    thumb_mcp_2 = (int(handL.landmark[2].x *width),int(handL.landmark[2].y *height))
    thumb_ip_2 = (int(handL.landmark[3].x *width),int(handL.landmark[3].y *height))
    thumb_tip_2 = (int(handL.landmark[4].x *width),int(handL.landmark[4].y *height))
    index_finger_mcp_2 = (int(handL.landmark[5].x *width),int(handL.landmark[5].y *height))
    index_finger_pip_2 = (int(handL.landmark[6].x *width),int(handL.landmark[6].y *height))
    index_finger_dip_2 = (int(handL.landmark[7].x *width),int(handL.landmark[7].y *height))
    index_finger_tip_2 = (int(handL.landmark[8].x *width),int(handL.landmark[8].y *height))
    middle_finger_mcp_2 = (int(handL.landmark[9].x *width),int(handL.landmark[9].y *height))
    middle_finger_pip_2 = (int(handL.landmark[10].x *width),int(handL.landmark[10].y *height))
    middle_finger_dip_2 = (int(handL.landmark[11].x *width),int(handL.landmark[11].y *height))
    middle_finger_tip_2 = (int(handL.landmark[12].x *width),int(handL.landmark[12].y *height))
    ring_finger_mcp_2 = (int(handL.landmark[13].x *width),int(handL.landmark[13].y *height))
    ring_finger_pip_2 = (int(handL.landmark[14].x *width),int(handL.landmark[14].y *height))
    ring_finger_dip_2 = (int(handL.landmark[15].x *width),int(handL.landmark[15].y *height))
    ring_finger_tip_2 = (int(handL.landmark[16].x *width),int(handL.landmark[16].y *height))
    pinky_finger_mcp_2 = (int(handL.landmark[17].x *width),int(handL.landmark[17].y *height))
    pinky_finger_pip_2 = (int(handL.landmark[18].x *width),int(handL.landmark[18].y *height))
    pinky_finger_dip_2 = (int(handL.landmark[19].x *width),int(handL.landmark[19].y *height))
    pinky_finger_tip_2 = (int(handL.landmark[20].x *width),int(handL.landmark[20].y *height))

    features = [spatial_distance(thumb_cmc_1, index_finger_mcp_1),
        spatial_distance(index_finger_mcp_1, middle_finger_mcp_1),
        spatial_distance(middle_finger_mcp_1, ring_finger_mcp_1),
        spatial_distance(ring_finger_mcp_1, pinky_finger_mcp_1),
        spatial_distance(thumb_mcp_1, index_finger_pip_1),
        spatial_distance(index_finger_pip_1, middle_finger_pip_1),
        spatial_distance(middle_finger_pip_1, ring_finger_pip_1),
        spatial_distance(ring_finger_pip_1, pinky_finger_pip_1),
        spatial_distance(thumb_ip_1, index_finger_dip_1),
        spatial_distance(index_finger_dip_1, middle_finger_dip_1),
        spatial_distance(middle_finger_dip_1, ring_finger_dip_1),
        spatial_distance(ring_finger_dip_1, pinky_finger_dip_1),
        spatial_distance(thumb_tip_1, index_finger_tip_1),
        spatial_distance(index_finger_tip_1, middle_finger_tip_1),
        spatial_distance(middle_finger_tip_1, ring_finger_tip_1),
        spatial_distance(ring_finger_tip_1, pinky_finger_tip_1),
        spatial_distance(thumb_cmc_2, index_finger_mcp_2),
        spatial_distance(index_finger_mcp_2, middle_finger_mcp_2),
        spatial_distance(middle_finger_mcp_2, ring_finger_mcp_2),
        spatial_distance(ring_finger_mcp_2, pinky_finger_mcp_2),
        spatial_distance(thumb_mcp_2, index_finger_pip_2),
        spatial_distance(index_finger_pip_2, middle_finger_pip_2),
        spatial_distance(middle_finger_pip_2, ring_finger_pip_2),
        spatial_distance(ring_finger_pip_2, pinky_finger_pip_2),
        spatial_distance(thumb_ip_2, index_finger_dip_2),
        spatial_distance(index_finger_dip_2, middle_finger_dip_2),
        spatial_distance(middle_finger_dip_2, ring_finger_dip_2),
        spatial_distance(ring_finger_dip_2, pinky_finger_dip_2),
        spatial_distance(thumb_tip_2, index_finger_tip_2),
        spatial_distance(index_finger_tip_2, middle_finger_tip_2),
        spatial_distance(middle_finger_tip_2, ring_finger_tip_2),
        spatial_distance(ring_finger_tip_2, pinky_finger_tip_2),
        ]
    
    return features



def get_features_1(hand,width,height):
    thumb_cmc_1 = (int(hand.landmark[1].x *width),int(hand.landmark[1].y *height))
    thumb_mcp_1 = (int(hand.landmark[2].x *width),int(hand.landmark[2].y *height))
    thumb_ip_1 = (int(hand.landmark[3].x *width),int(hand.landmark[3].y *height))
    thumb_tip_1 = (int(hand.landmark[4].x *width),int(hand.landmark[4].y *height))
    index_finger_mcp_1 = (int(hand.landmark[5].x *width),int(hand.landmark[5].y *height))
    index_finger_pip_1 = (int(hand.landmark[6].x *width),int(hand.landmark[6].y *height))
    index_finger_dip_1 = (int(hand.landmark[7].x *width),int(hand.landmark[7].y *height))
    index_finger_tip_1 = (int(hand.landmark[8].x *width),int(hand.landmark[8].y *height))
    middle_finger_mcp_1 = (int(hand.landmark[9].x *width),int(hand.landmark[9].y *height))
    middle_finger_pip_1 = (int(hand.landmark[10].x *width),int(hand.landmark[10].y *height))
    middle_finger_dip_1 = (int(hand.landmark[11].x *width),int(hand.landmark[11].y *height))
    middle_finger_tip_1 = (int(hand.landmark[12].x *width),int(hand.landmark[12].y *height))
    ring_finger_mcp_1 = (int(hand.landmark[13].x *width),int(hand.landmark[13].y *height))
    ring_finger_pip_1 = (int(hand.landmark[14].x *width),int(hand.landmark[14].y *height))
    ring_finger_dip_1 = (int(hand.landmark[15].x *width),int(hand.landmark[15].y *height))
    ring_finger_tip_1 = (int(hand.landmark[16].x *width),int(hand.landmark[16].y *height))
    pinky_finger_mcp_1 = (int(hand.landmark[17].x *width),int(hand.landmark[17].y *height))
    pinky_finger_pip_1 = (int(hand.landmark[18].x *width),int(hand.landmark[18].y *height))
    pinky_finger_dip_1 = (int(hand.landmark[19].x *width),int(hand.landmark[19].y *height))
    pinky_finger_tip_1 = (int(hand.landmark[20].x *width),int(hand.landmark[20].y *height))

    
    features = [spatial_distance(thumb_cmc_1, index_finger_mcp_1),
        spatial_distance(index_finger_mcp_1, middle_finger_mcp_1),
        spatial_distance(middle_finger_mcp_1, ring_finger_mcp_1),
        spatial_distance(ring_finger_mcp_1, pinky_finger_mcp_1),
        spatial_distance(thumb_mcp_1, index_finger_pip_1),
        spatial_distance(index_finger_pip_1, middle_finger_pip_1),
        spatial_distance(middle_finger_pip_1, ring_finger_pip_1),
        spatial_distance(ring_finger_pip_1, pinky_finger_pip_1),
        spatial_distance(thumb_ip_1, index_finger_dip_1),
        spatial_distance(index_finger_dip_1, middle_finger_dip_1),
        spatial_distance(middle_finger_dip_1, ring_finger_dip_1),
        spatial_distance(ring_finger_dip_1, pinky_finger_dip_1),
        spatial_distance(thumb_tip_1, index_finger_tip_1),
        spatial_distance(index_finger_tip_1, middle_finger_tip_1),
        spatial_distance(middle_finger_tip_1, ring_finger_tip_1),
        spatial_distance(ring_finger_tip_1, pinky_finger_tip_1),
        ]
    
    return features