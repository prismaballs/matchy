import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, StringVar, DoubleVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

try:
    # package import when installed or run as package
    from .matchy_logic import MatchyLogic, load_rew_txt
except Exception:
    # fallback to local import when running as a script from the repo folder
    from matchy_logic import MatchyLogic, load_rew_txt


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

        # Status message timer
        self._status_timer_id = None

        # Threading support
        self._worker_thread = None
        self._cancel_operation = False

        self._build_ui()

        # Setup proper cleanup on window close
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _thread_safe_call(self, func, *args, **kwargs):
        """Execute a function in the main thread safely."""
        self.after_idle(lambda: func(*args, **kwargs))

    def _is_operation_cancelled(self):
        """Check if the current operation should be cancelled."""
        return self._cancel_operation

    def _cancel_current_operation(self):
        """Cancel the current running operation."""
        self._cancel_operation = True

    def _reset_cancel_flag(self):
        """Reset the cancellation flag for new operations."""
        self._cancel_operation = False

    def _on_closing(self):
        """Handle application closing - cleanup threads."""
        self._cancel_current_operation()

        # Wait briefly for threads to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

        self.destroy()

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)
        self.tab_import, self.tab_prepare, self.tab_results = ttk.Frame(
            nb), ttk.Frame(nb), ttk.Frame(nb)
        nb.add(self.tab_import, text="Import")
        nb.add(self.tab_prepare, text="Prepare")
        nb.add(self.tab_results, text="Results")

        # Status label in the top-right corner of the notebook
        self.status_var = StringVar()
        self.status_label = ttk.Label(nb, textvariable=self.status_var,
                                      foreground='green', font=('TkDefaultFont', 9))
        self.status_label.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=5)

        self._build_import_tab(self.tab_import)
        self._build_prepare_tab(self.tab_prepare)
        self._build_results_tab(self.tab_results)

    def _set_loading_state(self, button, loading=True, elapsed_time=None):
        """Set loading state for a button with optional timing info and cancel option."""
        if loading:
            button.config(state='disabled', text='Processing...')
            self.update_idletasks()  # Force UI update
            # You could add a cancel button here if needed
        else:
            time_str = f" ({elapsed_time:.2f}s)" if elapsed_time is not None else ""
            if 'Next' in button['text'] or 'Processing' in button['text']:
                button.config(state='normal', text=f'Next{time_str}')
            elif 'Match' in button['text'] or 'Processing' in button['text']:
                button.config(state='normal', text=f'Match!{time_str}')
            else:
                button.config(state='normal')
            # Clear timing after 3 seconds
            if elapsed_time is not None:
                self.after(3000, lambda: self._clear_button_timing(button))

    def _clear_button_timing(self, button):
        """Clear timing information from button text."""
        if 'Next' in button['text']:
            button.config(text='Next')
        elif 'Match' in button['text']:
            button.config(text='Match!')

    def _show_status_message(self, message, duration=5000):
        """Show a status message at the top right that disappears after duration."""
        # Cancel any existing timer to prevent early clearing
        if hasattr(self, '_status_timer_id') and self._status_timer_id:
            self.after_cancel(self._status_timer_id)

        self.status_var.set(message)
        # Clear the message after the specified duration (if duration > 0)
        if duration > 0:
            self._status_timer_id = self.after(
                duration, lambda: self.status_var.set(""))

    def _show_loading_message(self, message):
        """Show a persistent loading message that won't auto-clear."""
        self._show_status_message(message, duration=0)

    def _clear_status_message(self):
        """Manually clear the status message."""
        if hasattr(self, '_status_timer_id') and self._status_timer_id:
            self.after_cancel(self._status_timer_id)
        self.status_var.set("")

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
        self.next_button = ttk.Button(
            right, text="Next", command=self.import_next_to_prepare)
        self.next_button.pack(fill=tk.X, pady=2)

    def select_folder(self):
        p = filedialog.askdirectory()
        if p:
            self.folder.set(p)
            self._scan_folder_threaded()

    def _scan_folder_threaded(self):
        """Scan folder in a background thread to avoid UI freezing."""
        if self._worker_thread and self._worker_thread.is_alive():
            return  # Already scanning

        self._reset_cancel_flag()
        self._show_loading_message("Scanning folder...")

        def scan_worker():
            start_time = time.time()  # Start timing inside the worker thread
            try:
                folder_path = self.folder.get()
                files = []
                file_stats = {}
                file_data_cache = {}

                txt_files = [fn for fn in sorted(os.listdir(folder_path))
                             if fn.lower().endswith('.txt')]

                for i, fn in enumerate(txt_files):
                    if self._is_operation_cancelled():
                        return

                    # Update progress with persistent message
                    progress_msg = f"Scanning files... {i+1}/{len(txt_files)}"
                    self._thread_safe_call(
                        self._show_loading_message, progress_msg)

                    f, y = load_rew_txt(os.path.join(folder_path, fn))
                    if f.size:
                        file_data_cache[fn] = (f, y)
                        files.append(fn)

                if not self._is_operation_cancelled():
                    # Calculate elapsed time after work is done
                    elapsed = time.time() - start_time

                    # Update UI in main thread
                    def update_ui():
                        self.files.clear()
                        self.file_stats.clear()
                        self.file_data_cache.clear()

                        self.files.extend(files)
                        self.file_stats.update(file_stats)
                        self.file_data_cache.update(file_data_cache)

                        self._refresh_file_tree(raw=True)
                        self._show_status_message(
                            f"Found {len(files)} files in {elapsed:.2f} seconds")

                    self._thread_safe_call(update_ui)

            except Exception as e:
                elapsed = time.time() - start_time
                error_msg = f"Error scanning folder: {str(e)}"
                self._thread_safe_call(self._show_status_message, error_msg)
                self._thread_safe_call(
                    messagebox.showerror, "Scan Error", error_msg)

        self._worker_thread = threading.Thread(target=scan_worker, daemon=True)
        self._worker_thread.start()

    def _scan_folder(self):
        """Legacy synchronous method - kept for compatibility."""
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
        """Process import settings in a background thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            return  # Already processing

        self._reset_cancel_flag()
        self._set_loading_state(self.next_button, loading=True)
        self._show_loading_message("Processing data...")

        def process_worker():
            start_time = time.time()  # Start timing inside the worker thread
            try:
                # Validate inputs first
                try:
                    fmin, fmax = float(self.fmin_var.get()), float(
                        self.fmax_var.get())
                except:
                    self._thread_safe_call(
                        messagebox.showerror, "Range", "Invalid frequency")
                    self._thread_safe_call(
                        self._set_loading_state, self.next_button, False)
                    return
                if fmin >= fmax:
                    self._thread_safe_call(
                        messagebox.showerror, "Range", "Min must be less than max")
                    self._thread_safe_call(
                        self._set_loading_state, self.next_button, False)
                    return

                if self._is_operation_cancelled():
                    return

                # Set up processing parameters
                self.freq_range = (fmin, fmax)
                ds_str = self.down_var.get()
                self.downsample = int(ds_str.split(
                    '/')[1]) if ds_str != 'none' else None

                # Update progress with persistent message
                self._thread_safe_call(
                    self._show_loading_message, "Applying import settings...")

                # Process data in background
                self.logic.file_data_cache = dict(self.file_data_cache)
                self.logic.files = list(self.files)

                if self._is_operation_cancelled():
                    return

                self.logic.apply_import_settings(
                    self.files, self.file_data_cache, fmin, fmax, self.downsample)

                if self._is_operation_cancelled():
                    return

                # Calculate elapsed time after work is done
                elapsed = time.time() - start_time

                # Update UI in main thread
                def update_ui():
                    self.processed_data_cache = self.logic.processed_data_cache
                    self.file_stats = self.logic.file_stats
                    self.filtered_files = self.logic.filtered_files
                    self.all_filtered_files = list(self.filtered_files)

                    self._refresh_file_tree(raw=False)
                    self.prepare_update_plot()
                    if hasattr(self, 'update_outlier_slider_range'):
                        self.update_outlier_slider_range()
                    self._on_outlier_change()

                    # Switch to prepare tab
                    self.nametowidget(self.winfo_children()[0]).select(1)

                    # Update button state with correct timing
                    self._set_loading_state(
                        self.next_button, loading=False, elapsed_time=elapsed)
                    self._show_status_message(
                        f"Processing finished in {elapsed:.2f} seconds")

                if not self._is_operation_cancelled():
                    self._thread_safe_call(update_ui)

            except Exception as e:
                elapsed = time.time() - start_time
                error_msg = f"Error processing data: {str(e)}"

                def handle_error():
                    self._set_loading_state(self.next_button, loading=False)
                    self._show_status_message("Processing failed")
                    messagebox.showerror("Processing Error", error_msg)

                self._thread_safe_call(handle_error)

        self._worker_thread = threading.Thread(
            target=process_worker, daemon=True)
        self._worker_thread.start()

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

        # Algorithm selection: placed to the right of the Outlier Tolerance frame
        alg_frame = ttk.Frame(bottom)
        alg_frame.pack(side=tk.LEFT, padx=8)
        ttk.Label(alg_frame, text="Algorithm:").pack(anchor='w')
        self.algorithm_var = tk.StringVar(value="default")
        ttk.Combobox(alg_frame, textvariable=self.algorithm_var, values=(
            "default", "blossom", "balanced"), state="readonly", width=12).pack(anchor='w', pady=4)

        self.metric_mode = tk.StringVar(value="rms")
        radio_frame = ttk.Frame(bottom)
        radio_frame.pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(radio_frame, text="RMS",
                        variable=self.metric_mode, value="rms").pack(side=tk.LEFT)
        ttk.Radiobutton(radio_frame, text="N/A",
                        variable=self.metric_mode, value="avg").pack(side=tk.LEFT)
        self.match_button = ttk.Button(bottom, text="Match!",
                                       command=self._apply_outliers_and_next)
        self.match_button.pack(side=tk.RIGHT)
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
        """Compute outlier data in a background thread if it's expensive."""
        if hasattr(self, "_outlier_data") and self._outlier_data is not None:
            return  # Already computed

        # For small datasets, compute synchronously
        if len(self.all_filtered_files) <= 20:
            self.logic.all_filtered_files = list(self.all_filtered_files)
            self.logic.processed_data_cache = dict(self.processed_data_cache)
            self.logic._compute_outlier_data()
            self._outlier_data = self.logic._outlier_data
            return

        # For larger datasets, use threading
        if self._worker_thread and self._worker_thread.is_alive():
            return  # Already computing

        self._reset_cancel_flag()
        self._show_loading_message("Computing outlier analysis...")

        def compute_worker():
            start_time = time.time()  # Start timing inside the worker thread
            try:
                self.logic.all_filtered_files = list(self.all_filtered_files)
                self.logic.processed_data_cache = dict(
                    self.processed_data_cache)

                if self._is_operation_cancelled():
                    return

                self.logic._compute_outlier_data()

                if not self._is_operation_cancelled():
                    # Calculate elapsed time after work is done
                    elapsed = time.time() - start_time

                    def update_ui():
                        self._outlier_data = self.logic._outlier_data
                        self._on_outlier_change_ui_update()
                        self._show_status_message(
                            f"Outlier analysis complete in {elapsed:.2f} seconds")

                    self._thread_safe_call(update_ui)

            except Exception as e:
                elapsed = time.time() - start_time
                error_msg = f"Error computing outliers: {str(e)}"
                self._thread_safe_call(self._show_status_message, error_msg)

        self._worker_thread = threading.Thread(
            target=compute_worker, daemon=True)
        self._worker_thread.start()

    def _on_outlier_change_ui_update(self):
        """Update UI components after outlier data changes - main thread only."""
        if not hasattr(self, "_outlier_data") or self._outlier_data is None:
            return
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

    def _on_outlier_change(self, _ev=None):
        if not hasattr(self, "_outlier_data") or self._outlier_data is None:
            self._compute_outlier_data()
            return  # UI will be updated by the worker thread

        # If outlier data is already available, update UI immediately
        self._on_outlier_change_ui_update()

    def _apply_outliers_and_next(self):
        """Apply outlier filtering and build partitions in a background thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            return  # Already processing

        self._reset_cancel_flag()
        self._set_loading_state(self.match_button, loading=True)
        self._show_loading_message("Building partitions...")

        def match_worker():
            start_time = time.time()  # Start timing inside the worker thread
            try:
                tol_rank_limit = int(self.out_tol.get())

                # Determine active files
                if not self._outlier_data:
                    active_files = list(self.all_filtered_files)
                else:
                    active_files = [fn for fn in self.all_filtered_files if self._outlier_data["filename_to_rank_map"].get(
                        fn, float('inf')) <= tol_rank_limit]

                if self._is_operation_cancelled():
                    return

                self._thread_safe_call(
                    self._show_loading_message, f"Computing partitions for {len(active_files)} files...")

                # Compute partitions in background thread (the heavy work)
                self.active_files = active_files

                # Get algorithm choice
                algo = getattr(self, 'algorithm_var',
                               None) and self.algorithm_var.get()

                # Do the heavy computation in the background thread
                try:
                    if algo == 'blossom':
                        strategies, model_rms_map = self.logic.build_partitions_with_algorithm(
                            active_files, "blossom")
                    elif algo == 'balanced':
                        strategies, model_rms_map = self.logic.build_partitions_with_algorithm(
                            active_files, "balanced")
                    else:
                        strategies, model_rms_map = self.logic.build_partitions_with_algorithm(
                            active_files, "default")
                except ImportError as ie:
                    # Handle missing dependencies
                    def show_import_error():
                        if 'networkx' in str(ie):
                            messagebox.showerror(
                                'Algorithm Error', "networkx is required for this algorithm; install with 'pip install networkx'")
                        else:
                            messagebox.showerror(
                                'Import Error', f"Missing dependency: {str(ie)}")
                        self._set_loading_state(
                            self.match_button, loading=False)
                    self._thread_safe_call(show_import_error)
                    return

                if self._is_operation_cancelled():
                    return

                # Calculate elapsed time after work is done
                elapsed = time.time() - start_time

                # Update UI in main thread with computed results
                def update_ui():
                    # Display the computed partitions
                    self._display_partitions_sync(
                        strategies, model_rms_map, active_files)

                    # Switch to results tab
                    self.nametowidget(self.winfo_children()[0]).select(2)

                    # Update button state with correct timing
                    self._set_loading_state(
                        self.match_button, loading=False, elapsed_time=elapsed)
                    self._show_status_message(
                        f"Matching finished in {elapsed:.2f} seconds")

                if not self._is_operation_cancelled():
                    self._thread_safe_call(update_ui)

            except Exception as e:
                elapsed = time.time() - start_time
                error_msg = f"Error building partitions: {str(e)}"

                def handle_error():
                    self._set_loading_state(self.match_button, loading=False)
                    self._show_status_message("Matching failed")
                    messagebox.showerror("Matching Error", error_msg)

                self._thread_safe_call(handle_error)

        self._worker_thread = threading.Thread(
            target=match_worker, daemon=True)
        self._worker_thread.start()

    def _goto_tab(self, idx): self.nametowidget(
        # Back to index 0 since we removed top_frame
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

    def _display_partitions_sync(self, strategies, model_rms_map, files):
        """Display computed partitions in the UI - main thread only."""
        tree = self.partition_tree
        for i in tree.get_children():
            tree.delete(i)
        self.ax2.clear()
        self.canvas2.draw_idle()

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
                pair_rms = model_rms_map.get(tuple(sorted(
                    pair)), 0) if model_rms_map else self.logic.calculate_deviation_from_processed(*pair)["rms"]
                pair_rms_str = f"{pair_rms:.4f}"
                tree.insert("", "end", values=(partition_id, avg_rms_str,
                            m1_name, m2_name, pair_rms_str, strat['leftover']))

    def _build_and_display_partitions(self, files):
        """Build partitions and update the UI. For large datasets, this may use threading internally."""
        tree = self.partition_tree
        for i in tree.get_children():
            tree.delete(i)
        self.ax2.clear()
        self.canvas2.draw_idle()

        # For small datasets, compute synchronously
        if len(files) <= 10:
            self._build_partitions_sync(files)
            return

        # For larger datasets, show progress and potentially use threading
        self._show_loading_message(
            f"Computing partitions for {len(files)} files...")

        # Use after_idle to prevent UI blocking during computation
        self.after_idle(lambda: self._build_partitions_sync(files))

    def _build_partitions_sync(self, files):
        """Synchronously build and display partitions."""
        tree = self.partition_tree

        # Choose algorithm based on dropdown selection
        algo = getattr(self, 'algorithm_var',
                       None) and self.algorithm_var.get()

        try:
            if algo == 'blossom':
                try:
                    strategies, model_rms_map = self.logic.build_partitions_with_algorithm(
                        files, "blossom")
                except ImportError:
                    messagebox.showerror(
                        'Blossom', "networkx is required for blossom; install with 'pip install networkx'")
                    return
            elif algo == 'balanced':
                try:
                    strategies, model_rms_map = self.logic.build_partitions_with_algorithm(
                        files, "balanced")
                except ImportError:
                    messagebox.showerror(
                        'Balanced', "networkx is required for balanced; install with 'pip install networkx'")
                    return
            else:
                strategies, model_rms_map = self.logic.build_partitions_with_algorithm(
                    files, "default")

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
                    pair_rms = model_rms_map.get(tuple(sorted(
                        pair)), 0) if model_rms_map else self.logic.calculate_deviation_from_processed(*pair)["rms"]
                    pair_rms_str = f"{pair_rms:.4f}"
                    tree.insert("", "end", values=(
                        partition_id, avg_rms_str, m1_name, m2_name, pair_rms_str, strat['leftover']))

        except Exception as e:
            error_msg = f"Error computing partitions: {str(e)}"
            messagebox.showerror("Partition Error", error_msg)
            tree.insert("", "end", values=(
                "(Error computing partitions)", "", "", "", "", ""))

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
