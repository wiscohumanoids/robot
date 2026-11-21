"""Select pick and place pixels and watch the simulated rollout."""

from PIL import Image, ImageTk


class InteractiveTrial:
    """Keep selection, live camera views, and results in one Tk window."""

    def __init__(self, rgb, blocks=(), *, workspace_outline=(), platform_outlines=(),
                 stacking=False):
        import tkinter as tk
        from tkinter import ttk

        height, width = rgb.shape[:2]
        self._tk = tk
        self._ttk = ttk
        self._root = tk.Tk()
        self._root.title("DINO pick and place")
        self._root.resizable(False, False)
        self._root.bind("<Escape>", lambda event: self.close())
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._closed = False

        self._source_prompt = (
            "Click a detected block to pick." if blocks else "Click the block to pick."
        )
        self._destination_prompt = "Click an empty spot to place the block."
        self._run_label = "Run pick and place"
        self._ready_prompt = "Click another destination or run pick and place."
        self._start_prompt = "Starting pick and place…"
        has_workspace_outline = len(workspace_outline) >= 3
        if has_workspace_outline:
            self._source_prompt += " The cyan outline shows the placement area."
            self._destination_prompt = "Click an empty spot inside the cyan outline to place the block."
        if stacking:
            self._source_prompt = "Click the block to move."
            self._destination_prompt = "Click the base block to stack onto."
            self._run_label = "Stack blocks"
            self._ready_prompt = "Click another base block or select Stack blocks."
            self._start_prompt = "Starting stack…"
            if len(blocks) > 2:
                self._source_prompt = "Click the first block to move."
                self._destination_prompt = "Click the block that will define the grid and stay on the bench."
        platforms = [polygon for polygon in platform_outlines if len(polygon) >= 3]
        if platforms:
            self._destination_prompt += " Gold outlines mark raised platforms."
        self._prompt = ttk.Label(self._root, text=self._source_prompt,
                                 wraplength=max(width - 16, 260), justify="center")
        self._prompt.pack(padx=8, pady=6)
        self._tracking_prompt = ttk.Label(self._root, justify="center")

        self._cameras = ttk.Frame(self._root)
        self._cameras.pack()
        self._overhead = self._create_camera("Overhead", rgb)
        self._wrist = None
        if has_workspace_outline:
            self._outline(workspace_outline, "cyan", "workspace", dash=(5, 3))
        for polygon in platforms:
            self._outline(polygon, "gold", "platform")
        for block in blocks:
            x0, y0, x1, y1 = block.bbox_xyxy
            self._overhead.create_rectangle(x0, y0, x1 - 1, y1 - 1, outline="white", tags="detection")
        self._buttons = ttk.Frame(self._root)
        self._buttons.pack(pady=6)
        self._close_button = ttk.Button(self._buttons, text="Cancel", command=self.close)
        self._close_button.pack(side="right", padx=4)

    @property
    def is_open(self):
        """Whether the trial can still display camera frames."""
        return not self._closed

    def _create_camera(self, name, rgb):
        panel = self._ttk.Frame(self._cameras)
        panel.pack(side="left", anchor="n", padx=4)
        self._ttk.Label(panel, text=name).pack(pady=(0, 4))
        canvas = self._tk.Canvas(panel, highlightthickness=0, borderwidth=0)
        canvas.pack()
        canvas.create_image(0, 0, anchor="nw", tags="frame")
        self._draw_frame(canvas, rgb)
        return canvas

    def _draw_frame(self, canvas, rgb):
        image = Image.fromarray(rgb)
        canvas.configure(width=image.width, height=image.height)
        # Tk holds the image name, so retain the Python image with its canvas.
        canvas.photo = ImageTk.PhotoImage(image, master=self._root)
        canvas.itemconfigure("frame", image=canvas.photo)

    def _outline(self, vertices, color, tag, **style):
        coordinates = [float(value) for vertex in vertices for value in vertex]
        self._overhead.create_polygon(*coordinates, fill="", outline=color, tags=tag, **style)

    def _mark(self, pixel, color, tag):
        x, y = map(float, pixel)
        self._overhead.create_oval(x - 5, y - 5, x + 5, y + 5, outline=color, width=2, tags=tag)

    def choose(self, validate_source, validate_destination):
        """Return accepted (x, y) source and destination pixels, or None on cancel.

        Validators receive the clicked pixel for the source and both pixels for
        the destination. Return a message to reject a click, or None to accept it.
        """
        if self._closed:
            return None
        canvas = self._overhead
        width, height = canvas.photo.width(), canvas.photo.height()
        source = destination = result = None
        callback_error = None

        def run():
            nonlocal result
            result = source, destination
            self._root.quit()

        run_button = self._ttk.Button(self._buttons, text=self._run_label, command=run,
                                      state="disabled")
        run_button.pack(side="left", padx=4)

        def reset():
            nonlocal source, destination
            source = destination = None
            canvas.delete("source", "destination", "rejected")
            run_button.state(["disabled"])
            reset_button.state(["disabled"])
            self._prompt.configure(text=self._source_prompt)

        reset_button = self._ttk.Button(self._buttons, text="Clear selection", command=reset,
                                        state="disabled")
        reset_button.pack(side="left", padx=4)

        def select(event):
            nonlocal source, destination
            x, y = event.x, event.y
            if not (0 <= x < width and 0 <= y < height):
                return
            pixel = x, y
            canvas.delete("rejected")
            if source is None:
                reason = validate_source(pixel)
                if reason is None:
                    source = pixel
                    self._mark(source, "lime", "source")
                    reset_button.state(["!disabled"])
                    self._prompt.configure(text=self._destination_prompt)
                else:
                    self._mark(pixel, "red", "rejected")
                    self._prompt.configure(text=reason)
                return

            canvas.delete("destination")
            destination = None
            run_button.state(["disabled"])
            reason = validate_destination(source, pixel)
            if reason is None:
                destination = pixel
                self._mark(destination, "magenta", "destination")
                run_button.state(["!disabled"])
                self._prompt.configure(text=self._ready_prompt)
            else:
                self._mark(pixel, "red", "rejected")
                self._prompt.configure(text=reason)

        def report_error(_type, value, traceback):
            nonlocal callback_error
            callback_error = value.with_traceback(traceback)
            self._root.quit()

        canvas.bind("<Button-1>", select)
        original_handler = self._root.report_callback_exception
        self._root.report_callback_exception = report_error
        try:
            self._root.mainloop()
        finally:
            self._root.report_callback_exception = original_handler
        if callback_error is not None:
            raise callback_error
        if self.is_open:
            canvas.unbind("<Button-1>")
            run_button.destroy()
            reset_button.destroy()
            self._close_button.configure(text="Close")
            self._prompt.configure(text=self._start_prompt)
        return result

    def tracking(self, source_pixel, destination_pixel, message):
        """Replace overhead markers with projected (x, y) pixels and show tracking status.

        Pass None for a point to clear its marker, or an empty message to hide
        the tracking status. The next camera refresh paints these changes.
        """
        if self._closed:
            return
        self._overhead.delete("source", "destination")
        if source_pixel is not None:
            self._mark(source_pixel, "lime", "source")
        if destination_pixel is not None:
            self._mark(destination_pixel, "magenta", "destination")
        if message:
            self._tracking_prompt.configure(text=message)
            self._tracking_prompt.pack(before=self._cameras, padx=8, pady=(0, 6))
        else:
            self._tracking_prompt.pack_forget()

    def show(self, overhead_rgb, wrist_rgb, phase):
        """Refresh both views and process pending Tk events on the main thread."""
        if self._closed:
            return
        if self._wrist is None:
            self._overhead.delete("workspace", "detection")
            self._wrist = self._create_camera("Active wrist", wrist_rgb)
        else:
            self._draw_frame(self._wrist, wrist_rgb)
        self._draw_frame(self._overhead, overhead_rgb)
        wraplength = max(overhead_rgb.shape[1] + wrist_rgb.shape[1] - 16, 260)
        self._prompt.configure(text=phase, wraplength=wraplength)
        self._tracking_prompt.configure(wraplength=wraplength)
        self._root.update()

    def finish(self, overhead_rgb, wrist_rgb, message):
        """Keep the final frames and outcome visible until the window closes."""
        self.show(overhead_rgb, wrist_rgb, message)
        if self.is_open:
            self._root.mainloop()

    def close(self):
        """Release Tk resources and let subsequent playback calls return immediately."""
        if self.is_open:
            self._closed = True
            self._root.destroy()
