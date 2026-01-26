"""
Sit-Up Exercise Monitor with GUI
Complete GUI application for sit-up monitoring
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
from SitUpExercise import SitUpMonitor


class SitUpGUI:
    """
    GUI Application for Sit-Up Exercise Monitoring
    """

    def __init__(self, root):
        """
        Initialize the GUI application
        :param root: Tkinter root window
        """
        self.root = root
        self.root.title("Sit-Up Exercise Monitor - Human Activity Monitoring")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2C3E50')

        # Variables
        self.monitor = SitUpMonitor()
        self.cap = None
        self.video_source = None
        self.is_running = False
        self.is_paused = False
        self.start_time = None
        self.video_thread = None

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        """
        Setup the GUI layout and widgets
        """
        # Title
        title_frame = tk.Frame(self.root, bg='#34495E', height=80)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(title_frame, text="SIT-UP EXERCISE MONITOR",
                               font=("Arial", 24, "bold"), bg='#34495E', fg='white')
        title_label.pack(pady=20)

        # Main container
        main_container = tk.Frame(self.root, bg='#2C3E50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left side - Video display
        left_frame = tk.Frame(main_container, bg='#2C3E50')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Video label
        self.video_label = tk.Label(left_frame, bg='black', text="No Video Source",
                                     font=("Arial", 16), fg='white')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Right side - Controls and Statistics
        right_frame = tk.Frame(main_container, bg='#34495E', width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        # Control Panel
        control_frame = tk.LabelFrame(right_frame, text="Control Panel",
                                      font=("Arial", 14, "bold"), bg='#34495E',
                                      fg='white', padx=10, pady=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Buttons
        btn_style = {'font': ("Arial", 11, "bold"), 'width': 18, 'height': 2}
        
        self.btn_camera = tk.Button(control_frame, text="📹 Open Camera",
                                     command=self.open_camera, bg='#3498DB',
                                     fg='white', **btn_style)
        self.btn_camera.pack(pady=5)

        self.btn_video = tk.Button(control_frame, text="📁 Open Video File",
                                    command=self.open_video, bg='#9B59B6',
                                    fg='white', **btn_style)
        self.btn_video.pack(pady=5)

        self.btn_start = tk.Button(control_frame, text="▶ Start",
                                    command=self.start_monitoring, bg='#27AE60',
                                    fg='white', state=tk.DISABLED, **btn_style)
        self.btn_start.pack(pady=5)

        self.btn_pause = tk.Button(control_frame, text="⏸ Pause",
                                    command=self.pause_monitoring, bg='#F39C12',
                                    fg='white', state=tk.DISABLED, **btn_style)
        self.btn_pause.pack(pady=5)

        self.btn_stop = tk.Button(control_frame, text="⏹ Stop",
                                   command=self.stop_monitoring, bg='#E74C3C',
                                   fg='white', state=tk.DISABLED, **btn_style)
        self.btn_stop.pack(pady=5)

        self.btn_reset = tk.Button(control_frame, text="🔄 Reset",
                                    command=self.reset_monitoring, bg='#95A5A6',
                                    fg='white', **btn_style)
        self.btn_reset.pack(pady=5)

        # Statistics Panel
        stats_frame = tk.LabelFrame(right_frame, text="Exercise Statistics",
                                    font=("Arial", 14, "bold"), bg='#34495E',
                                    fg='white', padx=10, pady=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        stats_style = {'font': ("Arial", 12), 'bg': '#34495E', 'fg': 'white', 'anchor': 'w'}

        # Count
        tk.Label(stats_frame, text="Sit-Ups Count:", **stats_style).pack(fill=tk.X, pady=2)
        self.lbl_count = tk.Label(stats_frame, text="0", font=("Arial", 28, "bold"),
                                   bg='#34495E', fg='#3498DB')
        self.lbl_count.pack(pady=5)

        # Time
        tk.Label(stats_frame, text="Elapsed Time:", **stats_style).pack(fill=tk.X, pady=2)
        self.lbl_time = tk.Label(stats_frame, text="00:00", font=("Arial", 18, "bold"),
                                  bg='#34495E', fg='#E67E22')
        self.lbl_time.pack(pady=5)

        # Current Angle
        tk.Label(stats_frame, text="Hip Angle:", **stats_style).pack(fill=tk.X, pady=2)
        self.lbl_angle = tk.Label(stats_frame, text="0°", font=("Arial", 18, "bold"),
                                   bg='#34495E', fg='#9B59B6')
        self.lbl_angle.pack(pady=5)

        # Form Quality
        tk.Label(stats_frame, text="Form Quality:", **stats_style).pack(fill=tk.X, pady=2)
        self.lbl_quality = tk.Label(stats_frame, text="0%", font=("Arial", 18, "bold"),
                                     bg='#34495E', fg='#27AE60')
        self.lbl_quality.pack(pady=5)

        # Good/Poor Form
        form_frame = tk.Frame(stats_frame, bg='#34495E')
        form_frame.pack(fill=tk.X, pady=10)

        tk.Label(form_frame, text="Good Form:", font=("Arial", 11),
                 bg='#34495E', fg='#2ECC71').pack(side=tk.LEFT)
        self.lbl_good = tk.Label(form_frame, text="0", font=("Arial", 11, "bold"),
                                  bg='#34495E', fg='#2ECC71')
        self.lbl_good.pack(side=tk.LEFT, padx=5)

        tk.Label(form_frame, text="Poor Form:", font=("Arial", 11),
                 bg='#34495E', fg='#E74C3C').pack(side=tk.LEFT, padx=(20, 0))
        self.lbl_poor = tk.Label(form_frame, text="0", font=("Arial", 11, "bold"),
                                  bg='#34495E', fg='#E74C3C')
        self.lbl_poor.pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please select a video source")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                             font=("Arial", 10), bg='#34495E', fg='white',
                             anchor='w', relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def open_camera(self):
        """
        Open the webcam for live monitoring
        """
        if self.is_running:
            messagebox.showwarning("Warning", "Please stop current session first!")
            return

        self.video_source = 0
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if self.cap.isOpened():
            self.status_var.set("Camera opened - Ready to start")
            self.btn_start.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Camera opened successfully!")
        else:
            self.status_var.set("Failed to open camera")
            messagebox.showerror("Error", "Failed to open camera!")

    def open_video(self):
        """
        Open a video file for analysis
        """
        if self.is_running:
            messagebox.showwarning("Warning", "Please stop current session first!")
            return

        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )

        if file_path:
            self.video_source = file_path
            self.cap = cv2.VideoCapture(file_path)

            if self.cap.isOpened():
                self.status_var.set(f"Video loaded: {file_path.split('/')[-1]}")
                self.btn_start.config(state=tk.NORMAL)
                messagebox.showinfo("Success", "Video loaded successfully!")
            else:
                self.status_var.set("Failed to load video")
                messagebox.showerror("Error", "Failed to load video file!")

    def start_monitoring(self):
        """
        Start the exercise monitoring
        """
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("Warning", "Please select a video source first!")
            return

        self.is_running = True
        self.is_paused = False
        self.start_time = time.time()
        self.status_var.set("Monitoring in progress...")

        # Update button states
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_camera.config(state=tk.DISABLED)
        self.btn_video.config(state=tk.DISABLED)

        # Start video processing thread
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()

    def pause_monitoring(self):
        """
        Pause/Resume the monitoring
        """
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="▶ Resume")
            self.status_var.set("Monitoring paused")
        else:
            self.btn_pause.config(text="⏸ Pause")
            self.status_var.set("Monitoring resumed")

    def stop_monitoring(self):
        """
        Stop the monitoring
        """
        self.is_running = False
        self.is_paused = False
        self.status_var.set("Monitoring stopped")

        # Update button states
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="⏸ Pause")
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_camera.config(state=tk.NORMAL)
        self.btn_video.config(state=tk.NORMAL)

        # Release video capture
        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.config(image='', text="No Video Source", bg='black')

    def reset_monitoring(self):
        """
        Reset all counters and statistics
        """
        self.monitor.reset()
        self.start_time = time.time() if self.is_running else None
        self.update_statistics()
        self.status_var.set("Statistics reset")
        messagebox.showinfo("Reset", "All statistics have been reset!")

    def process_video(self):
        """
        Process video frames in a separate thread
        """
        while self.is_running:
            if not self.is_paused:
                success, img = self.cap.read()

                if not success:
                    if self.video_source != 0:  # If video file ended
                        self.root.after(0, self.stop_monitoring)
                        self.root.after(0, lambda: messagebox.showinfo(
                            "Video Ended", "Video file has finished playing."))
                    break

                img = cv2.flip(img, 1)
                img = self.monitor.process_frame(img)

                # Update time
                if self.start_time:
                    self.monitor.elapsed_time = int(time.time() - self.start_time)

                # Update GUI
                self.root.after(0, self.update_video_display, img)
                self.root.after(0, self.update_statistics)

            time.sleep(0.01)

    def update_video_display(self, img):
        """
        Update the video display in GUI
        :param img: OpenCV image to display
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize to fit display
        display_width = 900
        display_height = int(img_pil.height * (display_width / img_pil.width))
        img_pil = img_pil.resize((display_width, display_height), Image.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(img_pil)
        self.video_label.config(image=img_tk, text='')
        self.video_label.image = img_tk

    def update_statistics(self):
        """
        Update statistics display
        """
        stats = self.monitor.get_statistics()
        
        self.lbl_count.config(text=str(stats['count']))
        
        minutes = stats['time'] // 60
        seconds = stats['time'] % 60
        self.lbl_time.config(text=f"{minutes:02d}:{seconds:02d}")
        
        self.lbl_angle.config(text=f"{stats['angle']}°")
        self.lbl_quality.config(text=f"{stats['quality']:.1f}%")
        self.lbl_good.config(text=str(stats['good_form']))
        self.lbl_poor.config(text=str(stats['poor_form']))

    def on_closing(self):
        """
        Handle window closing
        """
        if self.is_running:
            if messagebox.askokcancel("Quit", "Monitoring is in progress. Do you want to quit?"):
                self.is_running = False
                if self.cap:
                    self.cap.release()
                cv2.destroyAllWindows()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """
    Main function to run the application
    """
    root = tk.Tk()
    app = SitUpGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
