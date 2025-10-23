from itertools import combinations
import numpy as np
import os


def load_rew_txt(path):
    """Load a REW-style text file and return (freqs, spl)."""
    if not os.path.exists(path):
        return np.array([]), np.array([])
    freqs, vals = [], []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith("*"):
                    continue
                parts = ln.split()
                if len(parts) < 2:
                    continue
                try:
                    f, v = float(parts[0]), float(parts[1])
                    freqs.append(f)
                    vals.append(v)
                except (ValueError, IndexError):
                    continue
    except Exception:
        return np.array([]), np.array([])
    return np.array(freqs), np.array(vals)


def downsample_pairs(freqs, vals, factor):
    """Downsample by taking the first of each consecutive `factor` points."""
    if factor is None or factor <= 1:
        return freqs, vals
    return freqs[::int(factor)], vals[::int(factor)]


def calculate_deviation_from_pairs(f1, y1, f2, y2):
    """Given two frequency arrays and SPL arrays, return deviation metrics."""
    if f1 is None or f2 is None or y1 is None or y2 is None:
        return {"mean": 0, "rms": 0, "max": 0, "points": 0}
    if getattr(f1, 'size', 0) == 0 or getattr(f2, 'size', 0) == 0:
        return {"mean": 0, "rms": 0, "max": 0, "points": 0}
    # If axes identical, operate directly
    if np.array_equal(f1, f2):
        diff = np.abs(y1 - y2)
    else:
        common_f = np.linspace(max(f1.min(), f2.min()),
                               min(f1.max(), f2.max()), 2000)
        if common_f.size == 0:
            return {"mean": 0, "rms": 0, "max": 0, "points": 0}
        diff = np.abs(np.interp(common_f, f1, y1) -
                      np.interp(common_f, f2, y2))
    return {"mean": round(float(np.mean(diff)), 3), "rms": round(float(np.sqrt(np.mean(diff**2))), 3), "max": round(float(np.max(diff)), 3), "points": int(getattr(f1, 'size', len(f1)))}


def compute_pairwise_rms_map(files, processed_data_cache):
    """Compute a map { (a,b): rms } for all unordered pairs from `files`.

    Files are expected to be iterable of keys used in processed_data_cache. Missing
    entries fallback to empty arrays and yield an RMS of 0 via calculate_deviation_from_pairs
    (which returns 0 for empty inputs). The map keys are canonical tuples (sorted).
    """
    files = list(files)
    rms_map = {}
    for p in combinations(files, 2):
        a, b = p
        f1, y1 = processed_data_cache.get(a, (np.array([]), np.array([])))
        f2, y2 = processed_data_cache.get(b, (np.array([]), np.array([])))
        rms = calculate_deviation_from_pairs(f1, y1, f2, y2)['rms']
        rms_map[tuple(sorted((a, b)))] = rms
    return rms_map


def generate_all_pairings(items):
    """Yield all possible pairings for a list of items.

    Example: for [a,b,c,d] yields partitions like [(a,b),(c,d)], etc.
    """
    items = list(items)
    if len(items) < 2:
        yield []
        return
    p1 = items[0]
    rest = items[1:]
    for i in range(len(rest)):
        p2 = rest[i]
        remaining = rest[:i] + rest[i+1:]
        for sub_pairing in generate_all_pairings(remaining):
            yield [(p1, p2)] + sub_pairing


def build_partitions_enumeration(files, processed_data_cache):
    """Return a tuple (strategies, model_rms_map) where strategies is a list of dicts
    each containing 'avg_rms', 'partition', and 'leftover'.

    Uses exhaustive enumeration to find all possible partitions.
    """
    files = list(files)
    n = len(files)
    if n < 2:
        return [], {}
    # if n > 12:
    #     return [], {}

    # compute pair-wise RMS between all combinations
    model_rms_map = compute_pairwise_rms_map(files, processed_data_cache)

    all_strategies = []
    is_odd = n % 2 != 0

    groups_to_partition = [tuple(files)]
    leftover_map = {tuple(files): "None"}

    if is_odd:
        groups_to_partition = [c for c in combinations(files, n - 1)]
        full_set = set(files)
        leftover_map = {g: os.path.splitext(
            (full_set - set(g)).pop())[0] for g in groups_to_partition}

    for group in groups_to_partition:
        leftover = leftover_map.get(group, "None")
        unique_partitions_for_group = set()
        for partition_tuple in generate_all_pairings(group):
            canonical_partition = tuple(
                sorted([tuple(sorted(p)) for p in partition_tuple]))
            unique_partitions_for_group.add(canonical_partition)

        for partition in unique_partitions_for_group:
            rms_values = [model_rms_map.get(p, 999) for p in partition]
            if not rms_values:
                continue
            avg_rms = np.mean(rms_values)
            all_strategies.append(
                {'avg_rms': avg_rms, 'partition': partition, 'leftover': leftover})

    all_strategies.sort(key=lambda x: x['avg_rms'])
    return all_strategies, model_rms_map


def compute_outlier_data(processed_data_cache, all_filtered_files, metric_mode="rms"):
    """Compute outlier metadata used by the UI.

    Returns None if not computable, else a dict matching the previous shape.
    """
    all_curves = []
    for fn in all_filtered_files:
        f, y = processed_data_cache.get(fn, (np.array([]), np.array([])))
        if getattr(f, 'size', 0) > 0:
            all_curves.append({"filename": fn, "freqs": f, "spl": y})
    if len(all_curves) < 2:
        return None
    first_freq_axis = all_curves[0]['freqs']
    if not all(np.array_equal(first_freq_axis, c['freqs']) for c in all_curves):
        return None
    spl_matrix = np.array([c['spl'] for c in all_curves])
    n_files = len(all_curves)
    trim_count = n_files // 4
    if trim_count > 0 and n_files > 2 * trim_count:
        trimmed_mean_curve = np.mean(np.sort(spl_matrix, axis=0)[
                                     trim_count:-trim_count, :], axis=0)
    else:
        trimmed_mean_curve = np.mean(spl_matrix, axis=0)
    cumulative_dev = np.sum(spl_matrix - trimmed_mean_curve, axis=1)

    # Calculate deviation based on metric mode
    if metric_mode == "rms":
        # RMS deviation: sqrt(mean(squared_differences))
        squared_diff = (spl_matrix - trimmed_mean_curve) ** 2
        rms_dev_per_file = np.sqrt(np.mean(squared_diff, axis=1))
        dev_from_mean = np.abs(rms_dev_per_file - np.mean(rms_dev_per_file))
    else:  # metric_mode == "avg"
        # Average absolute deviation
        dev_from_mean = np.abs(cumulative_dev - np.mean(cumulative_dev))
    relative_curves = [{"filename": c["filename"], "freqs": c["freqs"], "relative": np.subtract(
        c["spl"], trimmed_mean_curve, out=np.ones_like(c["spl"]), where=trimmed_mean_curve != 0)} for c in all_curves]
    ranking = sorted(zip([c['filename'] for c in all_curves],
                     dev_from_mean), key=lambda x: x[1])
    return {"freqs": first_freq_axis, "all_curves": all_curves, "relative_curves": relative_curves, "trimmed_mean_curve": trimmed_mean_curve,
            "filename_to_rank_map": {fn: i + 1 for i, (fn, _) in enumerate(ranking)}, "filename_to_absdev_map": {fn: abs_dev for fn, abs_dev in ranking}}


def build_partitions_blossom(files, processed_data_cache):
    """Build partition strategies using a minimum-weight perfect matching (Blossom) approach.

    Returns (strategies, model_rms_map) where strategies is a list of dicts with keys
    'avg_rms', 'partition', 'leftover'. For even n this yields one best partition; for odd n
    we try each possible leftover and return the best partition for each leftover (so multiple
    strategies may be returned).
    """
    try:
        import networkx as nx
    except Exception:
        raise ImportError(
            "networkx is required for blossom matching. Install via 'pip install networkx'.")
    files = list(files)
    n = len(files)
    if n < 2:
        return [], {}
    # if n > 18:
    #     # safety: avoid extremely large calls; blossom is polynomial but trying many leftovers is costly
    #     return [], {}

    # compute pairwise RMS for all pairs
    model_rms_map = compute_pairwise_rms_map(files, processed_data_cache)

    strategies = []

    def matching_on_set(node_set):
        # build graph on node_set and compute min-sum matching by maximizing negative weights
        G = nx.Graph()
        for v in node_set:
            G.add_node(v)
        for a, b in combinations(node_set, 2):
            w = model_rms_map.get(tuple(sorted((a, b))), None)
            if w is None:
                continue
            # use negative RMS so that max_weight_matching finds minimum RMS
            G.add_edge(a, b, weight=-float(w))
        mate = nx.algorithms.matching.max_weight_matching(
            G, maxcardinality=True)
        # mate is a set of tuples; ensure canonical format
        partition = [tuple(sorted((u, v))) for u, v in mate]
        # compute average RMS for pairs
        if not partition:
            return None
        rms_vals = [model_rms_map.get(p, 999) for p in partition]
        avg_rms = float(np.mean(rms_vals)) if rms_vals else float('inf')
        return {'avg_rms': avg_rms, 'partition': tuple(partition), 'leftover': 'None'}

    if n % 2 == 0:
        res = matching_on_set(files)
        if res:
            strategies.append(res)
    else:
        # try each possible leftover and compute best matching on remaining nodes
        for leftover in files:
            remaining = [f for f in files if f != leftover]
            res = matching_on_set(remaining)
            if res:
                res2 = res.copy()
                res2['leftover'] = os.path.splitext(leftover)[0]
                strategies.append(res2)

    # sort strategies by avg_rms
    strategies.sort(key=lambda x: x['avg_rms'])
    return strategies, model_rms_map


def build_partitions_heuristic(files, processed_data_cache):
    """Build partitions using the original v0.3 heuristic algorithm.

    Returns (strategies, model_rms_map) where strategies is a list of dicts with keys
    'avg_rms', 'partition', 'leftover'.
    """
    files = list(files)
    n = len(files)
    if n < 2:
        return [], {}

    # compute pairwise RMS for all pairs
    model_rms_map = compute_pairwise_rms_map(files, processed_data_cache)

    # Build RMS diffs in the format expected by heuristic
    all_pairs = list(combinations(files, 2))
    rms_diffs_with_names = [
        (f"{f1}-{f2}", model_rms_map.get(tuple(sorted((f1, f2))), 999))
        for f1, f2 in all_pairs
    ]
    rms_diffs_with_names.sort(key=lambda x: x[1])

    # Core heuristic algorithm (inlined from partition_with_heuristic)
    if not rms_diffs_with_names or not files:
        return [], model_rms_map

    want_unmatched = 0 if n % 2 == 0 else 1
    pre_split = [(p, d, *p.split('-')) for p, d in rms_diffs_with_names]
    partitions = []

    for idx, (pair, diff, v1, v2) in enumerate(pre_split):
        inserted = False

        for part in partitions:
            m = part['matched']
            if v1 not in m and v2 not in m:
                m.update((v1, v2))
                part['unmatched'].difference_update((v1, v2))
                part['pairs'].append((pair, diff))
                part['score_sum'] += diff
                inserted = True

        if not inserted:
            prefix = pre_split[:idx]
            used = set()
            best_pairs, total_score = [], 0

            for p, d, a, b in prefix:
                if a in (v1, v2) or b in (v1, v2) or a in used or b in used:
                    continue
                used.update((a, b))
                best_pairs.append((p, d))
                total_score += d
                if len(used) >= n - (n % 2):
                    break

            matched = set(used)
            unmatched = set(files) - matched
            new_part = {
                'pairs': best_pairs[:],
                'matched': matched,
                'unmatched': unmatched,
                'score_sum': total_score
            }

            if v1 not in matched and v2 not in matched:
                new_part['pairs'].append((pair, diff))
                new_part['matched'].update((v1, v2))
                new_part['unmatched'].difference_update((v1, v2))
                new_part['score_sum'] += diff

            partitions.append(new_part)

        if sum(len(p['unmatched']) == want_unmatched for p in partitions) >= 3:
            break

    heuristic_partitions = sorted(
        partitions,
        key=lambda x: (len(x['unmatched']) != want_unmatched, x['score_sum'])
    )[:3]

    # Convert to the standard strategy format
    strategies = []
    for part in heuristic_partitions:
        num_pairs = len(part['pairs'])
        avg_rms = part['score_sum'] / num_pairs if num_pairs > 0 else 0

        partition_tuples = []
        for pair_str, _ in part['pairs']:
            f1, f2 = pair_str.split('-')
            partition_tuples.append(tuple(sorted((f1, f2))))

        leftover = "None"
        if part['unmatched']:
            unmatched_copy = part['unmatched'].copy()
            leftover = os.path.splitext(unmatched_copy.pop())[0]

        strategies.append({
            'avg_rms': avg_rms,
            'partition': tuple(partition_tuples),
            'leftover': leftover
        })

    return strategies, model_rms_map


def build_partitions_balanced(files, processed_data_cache, eps_list=None, centers=None):
    """Try to find pairings where pair RMS values are as even (low-variance) as possible.

    If no balanced perfect matching is found, fall back to Blossom exact matching.
    """
    try:
        import networkx as nx
    except Exception:
        raise ImportError(
            "networkx is required for balanced matching. Install via 'pip install networkx'.")

    files = list(files)
    n = len(files)
    if n < 2:
        return [], {}

    # compute pairwise RMS for all pairs
    model_rms_map = compute_pairwise_rms_map(files, processed_data_cache)

    if not model_rms_map:
        return [], {}

    rms_values = np.array(sorted(model_rms_map.values()))

    if centers is None:
        # Use statistical approach instead of trying every unique value
        centers = [
            np.min(rms_values),                    # Minimum (tight tolerance)
            np.percentile(rms_values, 25),         # 25th percentile
            np.median(rms_values),                 # Median (balanced)
            np.mean(rms_values),                   # Mean (average-focused)
            np.percentile(rms_values, 75),         # 75th percentile
            np.max(rms_values)                     # Maximum (loose tolerance)
        ]
        # Remove duplicates while preserving order
        centers = list(dict.fromkeys(centers))

    if eps_list is None:
        rms_range = np.max(rms_values) - np.min(rms_values)
        eps_list = [
            0.0,                           # Exact matches only
            rms_range * 0.05,             # Very tight (5% of range)
            rms_range * 0.1,              # Tight (10% of range)
            rms_range * 0.2,              # Moderate (20% of range)
            rms_range * 0.4,              # Loose (40% of range)
        ]

    edge_list = list(combinations(files, 2))
    edge_rms = np.array(
        [model_rms_map.get(tuple(sorted(edge)), np.inf) for edge in edge_list])

    strategies = []

    def best_matching_on_nodes(node_set):
        best = None
        node_set = list(node_set)
        node_set_set = set(node_set)  # For faster membership testing

        relevant_edges = []
        relevant_rms = []
        for i, (a, b) in enumerate(edge_list):
            if a in node_set_set and b in node_set_set:
                relevant_edges.append((a, b))
                relevant_rms.append(edge_rms[i])

        relevant_rms = np.array(relevant_rms)

        min_edges_needed = len(node_set) // 2

        for center in centers:
            best_for_center = None

            for eps in sorted(eps_list):
                low, high = center - eps, center + eps

                valid_mask = (relevant_rms >= low) & (relevant_rms <= high)
                valid_edge_count = np.sum(valid_mask)

                # Early skip if not enough edges for perfect matching
                if valid_edge_count < min_edges_needed:
                    continue

                G = nx.Graph()
                G.add_nodes_from(node_set)

                valid_indices = np.where(valid_mask)[0]
                for idx in valid_indices:
                    a, b = relevant_edges[idx]
                    G.add_edge(a, b, weight=1)

                # Quick check: ensure graph has enough connectivity
                if G.number_of_edges() < min_edges_needed:
                    continue

                try:
                    mate = nx.algorithms.matching.max_weight_matching(
                        G, maxcardinality=True)
                    if len(mate) * 2 != len(node_set):
                        continue

                    partition = [tuple(sorted(p)) for p in mate]
                    rms_vals = np.array([model_rms_map.get(p, 999)
                                        for p in partition])

                    mean = float(np.mean(rms_vals))
                    var = float(np.var(rms_vals))

                    candidate = {'avg_rms': mean, 'var': var,
                                 'partition': tuple(partition)}

                    # Update best for this center
                    if (best_for_center is None or
                        candidate['var'] < best_for_center['var'] - 1e-12 or
                        (abs(candidate['var'] - best_for_center['var']) < 1e-12 and
                         candidate['avg_rms'] < best_for_center['avg_rms'])):
                        best_for_center = candidate

                    if var < 0.01:  # Very low variance found
                        break

                except Exception:
                    continue

            # Update global best
            if best_for_center and (best is None or
                                    best_for_center['var'] < best['var'] - 1e-12 or
                                    (abs(best_for_center['var'] - best['var']) < 1e-12 and
                                        best_for_center['avg_rms'] < best['avg_rms'])):
                best = best_for_center

        return best

    if n % 2 == 0:
        best = best_matching_on_nodes(files)
        if best:
            best['leftover'] = 'None'
            strategies.append(best)
    else:
        for leftover in files:
            remaining = [f for f in files if f != leftover]
            best = best_matching_on_nodes(remaining)
            if best:
                best2 = best.copy()
                best2['leftover'] = os.path.splitext(leftover)[0]
                strategies.append(best2)

    # if no balanced strategies found, fall back to blossom
    if not strategies:
        try:
            strategies, _ = build_partitions_blossom(
                files, processed_data_cache)
        except Exception:
            # final fallback to enumeration
            strategies, _ = build_partitions_enumeration(
                files, processed_data_cache)

    # sort by variance then avg_rms if we have variance info
    strategies.sort(key=lambda x: (x.get('var', 0), x['avg_rms']))
    return strategies, model_rms_map


class MatchyLogic:
    """Encapsulates non-UI processing and calculations used by the app."""

    def __init__(self):
        self.file_data_cache = {}
        self.processed_data_cache = {}
        self.file_stats = {}
        self.files = []
        self.filtered_files = []
        self.all_filtered_files = []
        self.active_files = []
        self._outlier_data = None

    def compute_file_stats_from_cache(self, fn):
        f, y = self.file_data_cache.get(fn, (np.array([]), np.array([])))
        if f.size == 0:
            return
        self.file_stats[fn] = {'datapoints': len(f), 'min_freq': np.min(
            f), 'max_freq': np.max(f), 'avg_db': float(np.mean(y))}

    def apply_import_settings(self, files, file_data_cache, fmin, fmax, downsample=None):
        """Process raw loaded files and fill processed_data_cache and file_stats.

        Returns a tuple (processed_data_cache, file_stats, filtered_files).
        """
        self.file_data_cache = dict(file_data_cache)
        self.files = list(files)
        self.processed_data_cache = {}
        self.file_stats = {}
        self.filtered_files = []

        for fn in self.files:
            f, y = self.file_data_cache.get(fn, (np.array([]), np.array([])))
            if f.size == 0:
                continue
            mask = (f >= fmin) & (f <= fmax)
            f2, y2 = f[mask], y[mask]
            if f2.size == 0:
                continue
            if downsample:
                f2, y2 = downsample_pairs(f2, y2, downsample)
            self.processed_data_cache[fn] = (f2, y2)
            self.file_stats[fn] = {'datapoints': len(f2), 'min_freq': np.min(
                f2), 'max_freq': np.max(f2), 'avg_db': float(np.mean(y2))}
            self.filtered_files.append(fn)

        self.all_filtered_files = list(self.filtered_files)
        return self.processed_data_cache, self.file_stats, self.filtered_files

    def _compute_outlier_data(self, metric_mode="rms"):
        self._outlier_data = compute_outlier_data(
            self.processed_data_cache, self.all_filtered_files, metric_mode)

    def get_outlier_data(self, metric_mode="rms"):
        if not hasattr(self, '_outlier_data') or self._outlier_data is None:
            self._compute_outlier_data(metric_mode)
        return self._outlier_data

    def calculate_deviation_from_processed(self, f1_fn, f2_fn):
        """Calculate RMS deviation between two processed files.

        Returns a single float RMS value for compatibility with original v0.3.
        """
        f1, y1 = self.processed_data_cache.get(
            f1_fn, (np.array([]), np.array([])))
        f2, y2 = self.processed_data_cache.get(
            f2_fn, (np.array([]), np.array([])))
        if f1.size == 0 or f2.size == 0:
            return 0
        if np.array_equal(f1, f2):
            diff = np.abs(y1 - y2)
        else:
            f_min = max(f1.min(), f2.min())
            f_max = min(f1.max(), f2.max())
            common_f = np.linspace(f_min, f_max, 2000)
            if common_f.size == 0:
                return 0
            diff = np.abs(np.interp(common_f, f1, y1) -
                          np.interp(common_f, f2, y2))
        return float(np.sqrt(np.mean(diff**2)))

    def _generate_all_pairings(self, items):
        """Legacy method - now delegates to standalone function."""
        return generate_all_pairings(items)

    def build_partitions(self, files):
        """Return a list of partition strategies sorted by avg_rms.

        Each strategy is a dict: {'avg_rms', 'partition', 'leftover'}
        """
        strategies, _ = build_partitions_enumeration(
            files, self.processed_data_cache)
        return strategies

    def build_partitions_with_algorithm(self, files, algorithm="default"):
        """Build partitions using specified algorithm.

        Args:
            files: List of file names to partition
            algorithm: "default", "blossom", "balanced", or "heuristic"

        Returns:
            Tuple of (strategies, model_rms_map)
        """
        if algorithm == "blossom":
            return build_partitions_blossom(files, self.processed_data_cache)
        elif algorithm == "balanced":
            return build_partitions_balanced(files, self.processed_data_cache)
        elif algorithm == "heuristic":
            return build_partitions_heuristic(files, self.processed_data_cache)
        else:
            return build_partitions_enumeration(files, self.processed_data_cache)

    def get_pairwise_rms_map(self, files):
        """Get the pairwise RMS map for a set of files.

        Returns:
            Dictionary with (file1, file2) tuples as keys and RMS values as values
        """
        return compute_pairwise_rms_map(files, self.processed_data_cache)
