import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, StringVar, DoubleVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

try:
    # package import when installed or run as package
    from .matchy_logic import MatchyLogic, load_rew_txt, downsample_pairs
except Exception:
    # fallback to local import when running as a script from the repo folder
    from matchy_logic import MatchyLogic, load_rew_txt, downsample_pairs


class MatchyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Matchy v0.2.3")
        self.geometry("1000x600")

        # Logic handler
        self.logic = MatchyLogic()

        # --- State ---
        self.folder = StringVar(value="")
        self.files, self.file_stats, self.filtered_files, self.all_filtered_files, self.active_files = [], {}, [], [], []
        self.freq_range, self.downsample = (20.0, 20000.0), None

        # --- Caches ---
        self.file_data_cache, self.processed_data_cache, self._outlier_data = {}, {}, None

        self._build_ui()

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)
        self.tab_import, self.tab_prepare, self.tab_results = ttk.Frame(
            nb), ttk.Frame(nb), ttk.Frame(nb)
        nb.add(self.tab_import, text="Import")
        nb.add(self.tab_prepare, text="Prepare")
        nb.add(self.tab_results, text="Results")
        self._build_import_tab(self.tab_import)
        self._build_prepare_tab(self.tab_prepare)
        self._build_results_tab(self.tab_results)

    # ---------------- Import Tab ----------------
    def _build_import_tab(self, parent):
        left = ttk.Frame(parent)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        cols = ("Filename", "#datapoints", "Min_freq", "Max_freq")
        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.file_tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings")
        for c in cols:
            self.file_tree.heading(c, text=c)
            self.file_tree.column(c, width=80, anchor=tk.CENTER)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(parent, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
        ttk.Button(right, text="Select Folder",
                   command=self.select_folder).pack(fill=tk.X, pady=4)
        ttk.Label(right, textvariable=self.folder,
                  wraplength=320).pack(fill=tk.X, pady=4)
        hdr = ttk.LabelFrame(right, text="Preprocessing")
        hdr.pack(fill=tk.X, pady=6)
        fr = ttk.Frame(hdr)
        fr.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(fr, text="Freq min:").grid(row=0, column=0, sticky=tk.W)
        self.fmin_var = DoubleVar(value=20.0)
        ttk.Entry(fr, textvariable=self.fmin_var,
                  width=10).grid(row=0, column=1)
        ttk.Label(fr, text="Freq max:").grid(row=1, column=0, sticky=tk.W)
        self.fmax_var = DoubleVar(value=20000.0)
        ttk.Entry(fr, textvariable=self.fmax_var,
                  width=10).grid(row=1, column=1)
        ttk.Label(hdr, text="Downsample:").pack(anchor=tk.W)
        self.down_var = StringVar(value="none")
        ttk.Combobox(hdr, textvariable=self.down_var, values=(
            "none", "1/2", "1/3", "1/4"), state="readonly").pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(right, text="Next", command=self.import_next_to_prepare).pack(
            fill=tk.X, pady=2)

    def select_folder(self):
        p = filedialog.askdirectory()
        if p:
            self.folder.set(p)
            self._scan_folder()

    def _scan_folder(self):
        self.files.clear()
        self.file_stats.clear()
        self.file_data_cache.clear()
        for fn in sorted(os.listdir(self.folder.get())):
            if fn.lower().endswith('.txt'):
                f, y = load_rew_txt(os.path.join(self.folder.get(), fn))
                if f.size:
                    self.file_data_cache[fn] = (f, y)
                    self.files.append(fn)
        self._refresh_file_tree(raw=True)

    def _refresh_file_tree(self, raw=False):
        for i in self.file_tree.get_children():
            self.file_tree.delete(i)
        if not self.files:
            self.file_tree.insert("", "end", values=("(no files)", "", "", ""))
            return
        for fn in self.files:
            st = self.file_stats.get(fn, {}) if not raw else {}
            if raw:
                f, _ = self.file_data_cache.get(
                    fn, (np.array([]), np.array([])))
                if f.size:
                    st.update({'datapoints': len(f), 'min_freq': np.min(
                        f), 'max_freq': np.max(f)})
            self.file_tree.insert("", "end", values=(os.path.splitext(fn)[0], st.get(
                'datapoints', 0), f"{st.get('min_freq', 0):.3f}", f"{st.get('max_freq', 0):.3f}"))

    def apply_import_settings(self):
        try:
            fmin, fmax = float(self.fmin_var.get()), float(self.fmax_var.get())
        except:
            messagebox.showerror("Range", "Invalid frequency")
            return
        if fmin >= fmax:
            messagebox.showerror("Range", "Min must be less than max")
            return
        self.freq_range = (fmin, fmax)
        ds_str = self.down_var.get()
        self.downsample = int(ds_str.split(
            '/')[1]) if ds_str != 'none' else None
        self.logic.file_data_cache = dict(self.file_data_cache)
        self.logic.files = list(self.files)
        self.logic.apply_import_settings(
            self.files, self.file_data_cache, fmin, fmax, self.downsample)
        self.processed_data_cache = self.logic.processed_data_cache
        self.file_stats = self.logic.file_stats
        self.filtered_files = self.logic.filtered_files
        self.all_filtered_files = list(self.filtered_files)
        self._refresh_file_tree(raw=False)
        self.prepare_update_plot()
        if hasattr(self, 'update_outlier_slider_range'):
            self.update_outlier_slider_range()
        self._on_outlier_change()

    def import_next_to_prepare(self):
        self.apply_import_settings()
        self.nametowidget(self.winfo_children()[0]).select(1)
        self.prepare_update_plot()
        self._on_outlier_change()

    # ---------------- Prepare Tab ----------------
    def _build_prepare_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(top, width=500)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        cols = ("ID", "Filename", "datapoints", "Avg dB", "AbsDev", "Rank")
        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.prep_tree = ttk.Treeview(
            tree_frame, columns=cols, show='headings', selectmode='extended')
        for c in cols:
            self.prep_tree.heading(c, text=c)
            self.prep_tree.column(c, width=75, anchor=tk.CENTER)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self.prep_tree.yview)
        self.prep_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.prep_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._attach_sort_menu(self.prep_tree, {c: c for c in cols})
        self.show_relative_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Show graphs relative to curated mean",
                        variable=self.show_relative_var, command=self._on_outlier_change).pack(anchor='w', pady=4)
        bottom = ttk.Frame(parent)
        bottom.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(bottom, text="Back",
                   command=lambda: self._goto_tab(0)).pack(side=tk.LEFT)
        out_frame = ttk.LabelFrame(bottom, text="Outlier Tolerance")
        out_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=40, pady=5)
        out_frame.columnconfigure(1, weight=1)
        ttk.Label(out_frame, text="Keep Top:").grid(
            row=0, column=0, sticky='e', padx=5, pady=5)
        self.out_tol = tk.IntVar(value=1)
        self.outlier_scale = tk.Scale(out_frame, from_=1, to=2, orient='horizontal',
                                      variable=self.out_tol, command=self._on_outlier_change, resolution=1)
        self.outlier_scale.grid(row=0, column=1, sticky='ew', padx=5)
        self.out_tol_entry = ttk.Entry(
            out_frame, width=6, textvariable=self.out_tol)
        self.out_tol_entry.grid(row=0, column=2, padx=5)
        self.out_tol_entry.bind(
            "<Return>", lambda e: self._on_outlier_change())
        self.update_outlier_slider_range = lambda: (self.outlier_scale.configure(from_=1, to=max(
            1, len(self.all_filtered_files))), self.out_tol.set(max(1, len(self.all_filtered_files))))
        self.metric_mode = tk.StringVar(value="rms")
        radio_frame = ttk.Frame(bottom)
        radio_frame.pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(radio_frame, text="RMS",
                        variable=self.metric_mode, value="rms").pack(side=tk.LEFT)
        ttk.Radiobutton(radio_frame, text="N/A",
                        variable=self.metric_mode, value="avg").pack(side=tk.LEFT)
        ttk.Button(bottom, text="Match!",
                   command=self._apply_outliers_and_next).pack(side=tk.RIGHT)
        plot_frame = ttk.Frame(top)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=8)
        self.fig = Figure(figsize=(6, 3))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def prepare_update_plot(self):
        for i in self.prep_tree.get_children():
            self.prep_tree.delete(i)
        for idx, fn in enumerate(self.all_filtered_files, start=1):
            st = self.file_stats.get(fn, {})
            self.prep_tree.insert('', 'end', iid=fn, values=(idx, os.path.splitext(
                fn)[0], st.get('datapoints', 0), f"{st.get('avg_db', 0.0):.2f}", "", ""))
        self._plot_prep()

    def _plot_prep(self):
        self.ax.clear()
        if not self.all_filtered_files:
            self.ax.set_title('(no files)')
            self.canvas.draw_idle()
            return
        for fn in self.all_filtered_files[:6]:
            f, y = self.processed_data_cache.get(
                fn, (np.array([]), np.array([])))
            if f.size:
                self.ax.plot(f, y, label=os.path.splitext(fn)
                             [0], linewidth=0.6)
        self.ax.set_xscale('log')
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('SPL (dB)')
        self.ax.grid(True, which='both', linestyle='--',
                     linewidth=0.4, alpha=0.7)
        self.canvas.draw_idle()

    def _compute_outlier_data(self):
        # delegate to logic
        self.logic.all_filtered_files = list(self.all_filtered_files)
        self.logic.processed_data_cache = dict(self.processed_data_cache)
        self.logic._compute_outlier_data()
        self._outlier_data = self.logic._outlier_data

    def _on_outlier_change(self, _ev=None):
        if not hasattr(self, "_outlier_data") or self._outlier_data is None:
            self._compute_outlier_data()
        if not self._outlier_data:
            return
        data = self._outlier_data
        tol_rank_limit = int(self.out_tol.get())
        for fn in self.all_filtered_files:
            if not self.prep_tree.exists(fn):
                continue
            rank, abs_dev = data["filename_to_rank_map"].get(
                fn), data["filename_to_absdev_map"].get(fn)
            if rank is None:
                continue
            self.prep_tree.item(
                fn, tags=('ok' if rank <= tol_rank_limit else 'out',))
            self.prep_tree.set(fn, column='AbsDev', value=f"{abs_dev:.3f}")
            self.prep_tree.set(fn, column='Rank', value=str(rank))
        self.prep_tree.tag_configure(
            'out', background='#e0e0e0', foreground='gray')
        self.prep_tree.tag_configure('ok', background='', foreground='')
        self.ax.clear()
        show_relative = self.show_relative_var.get()
        if show_relative:
            self.ax.set_ylabel('dB relative to curated mean')
            self.ax.set_title('Relative graphs vs. curated mean')
            self.ax.axhline(0.0, color='#39FF14',
                            linestyle='--', linewidth=2, zorder=3)
            curves_to_plot, y_key = data["relative_curves"], "relative"
        else:
            self.ax.set_ylabel('SPL (dB)')
            self.ax.set_title('Frequency Response (Outliers Grayed)')
            self.ax.plot(data["freqs"], data["trimmed_mean_curve"], label='50% Mean',
                         linestyle='--', linewidth=2, color='#39FF14', zorder=3)
            curves_to_plot, y_key = data["all_curves"], "spl"
        for curve in curves_to_plot:
            rank = data["filename_to_rank_map"].get(
                curve["filename"], float('inf'))
            is_outlier = rank > tol_rank_limit
            self.ax.plot(curve["freqs"], curve[y_key], label=os.path.splitext(curve["filename"])[
                         0], linewidth=1, color='gray' if is_outlier else None, alpha=0.2 if is_outlier else 1.0)
        self.ax.set_xscale('log')
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.grid(True, which='both', linestyle='--',
                     linewidth=0.4, alpha=0.7)
        self.canvas.draw_idle()

    def _apply_outliers_and_next(self):
        tol_rank_limit = int(self.out_tol.get())
        if not self._outlier_data:
            self.active_files = list(self.all_filtered_files)
        else:
            self.active_files = [fn for fn in self.all_filtered_files if self._outlier_data["filename_to_rank_map"].get(
                fn, float('inf')) <= tol_rank_limit]
        self._build_and_display_partitions(self.active_files)
        self.nametowidget(self.winfo_children()[0]).select(2)

    def _goto_tab(self, idx): self.nametowidget(
        self.winfo_children()[0]).select(idx)

    # ---------------- Results Tab ----------------
    def _build_results_tab(self, parent):
        main_pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_frame = ttk.LabelFrame(main_pane, text="Monitor Partitions")
        main_pane.add(left_frame, weight=3)

        cols = ("Partition ID", "Partition Avg RMS",
                "Monitor 1", "Monitor 2", "Pair RMS", "Leftover")
        self.partition_tree = ttk.Treeview(
            left_frame, columns=cols, show='headings')
        for c in cols:
            w = 120 if c in ("Monitor 1", "Monitor 2") else 100
            self.partition_tree.heading(c, text=c)
            self.partition_tree.column(c, width=w, anchor=tk.CENTER)

        vsb = ttk.Scrollbar(left_frame, orient="vertical",
                            command=self.partition_tree.yview)
        self.partition_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.partition_tree.pack(fill=tk.BOTH, expand=True)
        self._attach_sort_menu(self.partition_tree, {})
        self.partition_tree.bind('<<TreeviewSelect>>',
                                 self._on_partition_select)

        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=2)
        self.fig2 = Figure(figsize=(6, 5))
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=right_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        ttk.Button(right_frame, text="Export CSV", command=self.export_list).pack(
            side=tk.BOTTOM, fill=tk.X, pady=4)

    def _build_and_display_partitions(self, files):
        tree = self.partition_tree
        for i in tree.get_children():
            tree.delete(i)
        self.ax2.clear()
        self.canvas2.draw_idle()

        strategies = self.logic.build_partitions(files)

        if not strategies:
            if len(files) < 2:
                tree.insert("", "end", values=(
                    "(Need at least 2 monitors)", "", "", "", "", ""))
            else:
                tree.insert("", "end", values=(
                    f"({len(files)} is too many for analysis)", "", "", "", "", ""))
            return

        for i, strat in enumerate(strategies):
            partition_id = i + 1
            avg_rms_str = f"{strat['avg_rms']:.4f}"
            for pair in strat['partition']:
                m1_name = os.path.splitext(pair[0])[0]
                m2_name = os.path.splitext(pair[1])[0]
                pair_rms = self.logic.calculate_deviation_from_processed(
                    *pair)["rms"]
                pair_rms_str = f"{pair_rms:.4f}"
                tree.insert("", "end", values=(partition_id, avg_rms_str,
                            m1_name, m2_name, pair_rms_str, strat['leftover']))

    def _on_partition_select(self, event):
        tree = self.partition_tree
        sel = tree.selection()
        if not sel:
            return

        vals = tree.item(sel[0], 'values')
        if len(vals) < 4:
            return

        m1_name, m2_name = vals[2], vals[3]
        fns = [fn for fn in self.all_filtered_files if os.path.splitext(fn)[
            0] in (m1_name, m2_name)]
        if len(fns) == 2:
            self.plot_pair(fns[0], fns[1])

    def plot_pair(self, f1_basename, f2_basename):
        self.ax2.clear()
        f1p, y1p = self.processed_data_cache.get(
            f1_basename, (np.array([]), np.array([])))
        f2p, y2p = self.processed_data_cache.get(
            f2_basename, (np.array([]), np.array([])))
        if f1p.size:
            self.ax2.plot(f1p, y1p, label=os.path.splitext(
                f1_basename)[0], linewidth=0.8)
        if f2p.size:
            self.ax2.plot(f2p, y2p, label=os.path.splitext(
                f2_basename)[0], linewidth=0.8)
        self.ax2.set_xscale('log')
        self.ax2.set_title(
            f"Pair: {os.path.splitext(f1_basename)[0]} vs {os.path.splitext(f2_basename)[0]}")
        self.ax2.legend(loc='best')
        self.ax2.grid(True, which='both', linestyle='--',
                      linewidth=0.4, alpha=0.7)
        self.canvas2.draw_idle()

    def export_list(self):
        if not self.partition_tree.get_children():
            messagebox.showwarning("Export", "No data to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[
                                            ("CSV files", "*.csv")], initialfile="matchy_partitions.csv")
        if not path:
            return
        try:
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                writer.writerow(self.partition_tree["columns"])
                for iid in self.partition_tree.get_children(''):
                    writer.writerow(self.partition_tree.item(iid, 'values'))
            messagebox.showinfo("Export CSV", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Export CSV", f"Failed to save:\n{e}")

    def _attach_sort_menu(self, tree, headers_dict):
        menu = tk.Menu(self, tearoff=0)

        def do_sort(col, reverse=False):
            current_cols = tree['columns']
            if col not in current_cols:
                return

            items = [(tree.set(k, col), k) for k in tree.get_children('')]
            try:
                items = [(float(v), k) for v, k in items if v]
            except (ValueError, TypeError):
                pass
            if len(set(v for v, _ in items)) <= 1:
                return

            items.sort(reverse=reverse)
            for i, (_, k) in enumerate(items):
                tree.move(k, '', i)

            for c in current_cols:
                tree.heading(c, text=c)
            tree.heading(col, text=f"★ {col} {'↓' if reverse else '↑'}")

        def popup(event):
            col_id_str = tree.identify_column(event.x)
            if not col_id_str.startswith("#"):
                return
            col_idx = int(col_id_str[1:]) - 1
            if col_idx < 0 or col_idx >= len(tree["columns"]):
                return
            colid = tree["columns"][col_idx]
            menu.delete(0, "end")
            menu.add_command(
                label=f"Sort '{colid}' Asc", command=lambda: do_sort(colid, False))
            menu.add_command(
                label=f"Sort '{colid}' Desc", command=lambda: do_sort(colid, True))
            menu.tk_popup(event.x_root, event.y_root)
        tree.bind("<Button-3>", popup)
