import pytest
from detector.tracker import SimpleTracker


def test_first_frame_assigns_ids():
    """第一帧应该给每头牛分配唯一ID"""
    tracker = SimpleTracker()
    detections = [
        [100, 100, 300, 300, 0.9, 'cow'],
        [400, 200, 600, 450, 0.8, 'cow'],
    ]
    tracks = tracker.update(detections)
    assert len(tracks) == 2
    assert tracks[0][4] == 1  # 第一头牛ID=1
    assert tracks[1][4] == 2  # 第二头牛ID=2


def test_same_cow_keeps_id():
    """同一头牛在下一帧应该保持同一个ID"""
    tracker = SimpleTracker()

    # 第一帧
    detections1 = [[100, 100, 300, 300, 0.9, 'cow']]
    tracker.update(detections1)

    # 第二帧，牛移动了一点点
    detections2 = [[105, 105, 305, 305, 0.9, 'cow']]
    tracks = tracker.update(detections2)

    assert tracks[0][4] == 1  # 还是ID=1


def test_new_cow_gets_new_id():
    """新出现的牛应该得到新ID"""
    tracker = SimpleTracker()

    # 第一帧：一头牛
    detections1 = [[100, 100, 300, 300, 0.9, 'cow']]
    tracker.update(detections1)

    # 第二帧：原来的牛 + 新的牛
    detections2 = [
        [105, 105, 305, 305, 0.9, 'cow'],  # 原来的牛
        [500, 500, 700, 700, 0.8, 'cow'],  # 新的牛
    ]
    tracks = tracker.update(detections2)

    ids = [t[4] for t in tracks]
    assert 1 in ids  # 原来的牛还是ID=1
    assert 2 in ids  # 新的牛是ID=2


def test_disappeared_cow_lost_count():
    """消失的牛lost计数应该增加"""
    tracker = SimpleTracker()

    # 第一帧：2头牛
    detections1 = [
        [100, 100, 300, 300, 0.9, 'cow'],
        [400, 400, 600, 600, 0.8, 'cow'],
    ]
    tracker.update(detections1)

    # 第二帧：只有一头牛
    detections2 = [[105, 105, 305, 305, 0.9, 'cow']]
    tracker.update(detections2)

    # 消失的牛lost应该是1
    lost_tracks = [t for t in tracker.tracks if t['lost'] > 0]
    assert len(lost_tracks) == 1
    assert lost_tracks[0]['lost'] == 1