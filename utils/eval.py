#!/usr/bin/python3

import numpy as np
import torch
import torch.nn.functional as F
import copy


class Scorer(object):

    def __init__(self):
        self.total_frames = 0
        self.ce_score = 0
        self.correct_frames = 0

    def add_sequence(self, scores, labels):
        scores = torch.FloatTensor(scores)
        labels = torch.LongTensor(labels)
        self.ce_score += F.nll_loss(scores, labels, reduction = 'sum').numpy()
        self.correct_frames += torch.sum(scores.max(dim = 1)[1] == labels).numpy()
        self.total_frames += labels.shape[0]

    def cross_entropy(self):
        return self.ce_score / self.total_frames

    def frame_accuracy(self):
        return self.correct_frames / self.total_frames


class MapScorer(object):

    class Segment(object):
    
        def __init__(self, start, end, score = 0):
            self.start = start
            self.end = end
            self.score = score
            self.is_hit = False

        def overlap(self, other):
            left = max(self.start, other.start) 
            right = min(self.end, other.end)
            if left > right:
                return 0
            else:
                return (right - left + 1) / (max(self.end, other.end) - min(self.start, other.start) + 1)

    def __init__(self):
        self.predicted_segments = dict()
        self.ground_truth_segments = dict()

    def _add_segments(self, segments, labels, scores = None):
        prev_label = 0
        for i, l in enumerate(labels):
            if l != prev_label and l > 0:
                segments[l] = segments.get(l, []) + [self.Segment(i, i, 0 if scores is None else scores[i])]
            elif l == prev_label and l > 0:
                segments[l][-1].end = i
                segments[l][-1].score += 0 if scores is None else scores[i]
            prev_label = l

    def _normalize_scores(self, segments):
        for segment in segments:
            segment.score = segment.score / (segment.end - segment.start + 1)

    def _rank_segments(self, ground_truth, prediction, threshold):
        self._normalize_scores(prediction)
        prediction = sorted(prediction, key=lambda segment: segment.score, reverse=True)
        ranked_segments = []
        for segment in prediction:
            best_match = None
            best_overlap = 0
            for gt in ground_truth:
                if segment.overlap(gt) >= best_overlap and segment.overlap(gt) >= threshold:
                    best_match = gt
                    best_overlap = segment.overlap(gt)
            if not best_match is None:
                segment.is_hit = True
                ground_truth.remove(best_match)
            ranked_segments.append(segment)
        return ranked_segments

    def _mAP(self, ground_truth, prediction, threshold):
        n_gt_segments = len(ground_truth)
        ranked_segments = self._rank_segments(ground_truth, prediction, threshold)
        hits = [ 1 if segment.is_hit else 0 for segment in ranked_segments ]
        prec = [ sum(hits[:k]) / k for k in range(1, len(hits)+1) ]
        ap = sum([prec[k] * hits[k] for k in range(len(hits))]) / n_gt_segments
        return ap

    def add_sequence(self, scores, labels):
        self._add_segments(self.ground_truth_segments, labels)
        self._add_segments(self.predicted_segments, scores.argmax(axis=1), scores.max(axis=1))

    def mAP(self, threshold):
        mAP = 0
        for label in self.ground_truth_segments:
            ground_truth = copy.deepcopy(self.ground_truth_segments[label])
            prediction = copy.deepcopy(self.predicted_segments[label]) if label in self.predicted_segments else []
            class_mAP = self._mAP(ground_truth, prediction, threshold)
            mAP += class_mAP
        return mAP / len(self.ground_truth_segments)


def thumos_output(filename, scores):
    prev_label = -1
    labels = np.argmax(scores, axis = 1)
    segments = []
    for i, l in enumerate(labels):
        if l != prev_label:
            segments += [ [l, 1, scores[i, l]] ]
        else:
            segments[-1][1] += 1
            segments[-1][2] += scores[i, l]
        prev_label = l
    for segment in segments:
        segment[2] = segment[2] / segment[1] # score normalization by segment length
    with open(filename, 'w') as f:
        f.write(' '.join( [ '%d:%d:%.4f' % (seg[0], seg[1], seg[2]) for seg in segments ] ) + '\n')

