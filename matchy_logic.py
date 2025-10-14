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

    def _compute_outlier_data(self):
        all_curves = []
        for fn in self.all_filtered_files:
            f, y = self.processed_data_cache.get(
                fn, (np.array([]), np.array([])))
            if f.size > 0:
                all_curves.append({"filename": fn, "freqs": f, "spl": y})
        if len(all_curves) < 2:
            self._outlier_data = None
            return
        first_freq_axis = all_curves[0]['freqs']
        if not all(np.array_equal(first_freq_axis, c['freqs']) for c in all_curves):
            self._outlier_data = None
            return
        spl_matrix = np.array([c['spl'] for c in all_curves])
        n_files = len(all_curves)
        trim_count = n_files // 4
        if trim_count > 0 and n_files > 2*trim_count:
            trimmed_mean_curve = np.mean(np.sort(spl_matrix, axis=0)[
                                         trim_count:-trim_count, :], axis=0)
        else:
            trimmed_mean_curve = np.mean(spl_matrix, axis=0)
        cumulative_dev = np.sum(spl_matrix - trimmed_mean_curve, axis=1)
        abs_dev_from_mean = np.abs(cumulative_dev - np.mean(cumulative_dev))
        relative_curves = [{"filename": c["filename"], "freqs": c["freqs"], "relative": np.subtract(
            c["spl"], trimmed_mean_curve, out=np.ones_like(c["spl"]), where=trimmed_mean_curve != 0)} for c in all_curves]
        ranking = sorted(
            zip([c['filename'] for c in all_curves], abs_dev_from_mean), key=lambda x: x[1])
        self._outlier_data = {"freqs": first_freq_axis, "all_curves": all_curves, "relative_curves": relative_curves, "trimmed_mean_curve": trimmed_mean_curve,
                              "filename_to_rank_map": {fn: i + 1 for i, (fn, _) in enumerate(ranking)}, "filename_to_absdev_map": {fn: abs_dev for fn, abs_dev in ranking}}

    def get_outlier_data(self):
        if not hasattr(self, '_outlier_data') or self._outlier_data is None:
            self._compute_outlier_data()
        return self._outlier_data

    def calculate_deviation_from_processed(self, f1_fn, f2_fn):
        f1, y1 = self.processed_data_cache.get(
            f1_fn, (np.array([]), np.array([])))
        f2, y2 = self.processed_data_cache.get(
            f2_fn, (np.array([]), np.array([])))
        if f1.size == 0 or f2.size == 0:
            return {"mean": 0, "rms": 0, "max": 0, "points": 0}
        if np.array_equal(f1, f2):
            diff = np.abs(y1 - y2)
        else:
            common_f = np.linspace(
                max(f1.min(), f2.min()), min(f1.max(), f2.max()), 2000)
            if common_f.size == 0:
                return {"mean": 0, "rms": 0, "max": 0, "points": 0}
            diff = np.abs(np.interp(common_f, f1, y1) -
                          np.interp(common_f, f2, y2))
        return {"mean": round(float(np.mean(diff)), 3), "rms": round(float(np.sqrt(np.mean(diff**2))), 3), "max": round(float(np.max(diff)), 3), "points": len(f1)}

    def _generate_all_pairings(self, items):
        items = list(items)
        if len(items) < 2:
            yield []
            return
        p1 = items[0]
        rest = items[1:]
        for i in range(len(rest)):
            p2 = rest[i]
            remaining = rest[:i] + rest[i+1:]
            for sub_pairing in self._generate_all_pairings(remaining):
                yield [(p1, p2)] + sub_pairing

    def build_partitions(self, files):
        """Return a list of partition strategies sorted by avg_rms.

        Each strategy is a dict: {'avg_rms', 'partition', 'leftover'}
        """
        n = len(files)
        if n < 2:
            return []
        if n > 12:
            return []

        model_rms_map = {tuple(sorted(p)): self.calculate_deviation_from_processed(
            *p)["rms"] for p in combinations(files, 2)}

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
            for partition_tuple in self._generate_all_pairings(group):
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
        return all_strategies
