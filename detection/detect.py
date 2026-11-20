from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "weights" / "best.pt"


class DetectUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Detect UI")
        self.root.geometry("760x260")

        self.model_path = tk.StringVar(value=str(DEFAULT_MODEL))
        self.source_path = tk.StringVar(value="")
        self.source_type = tk.StringVar(value="image")
        self.conf_value = tk.StringVar(value="0.25")
        self.status_text = tk.StringVar(value="请选择图片或文件夹。")

        self.build_ui()

    def build_ui(self):
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="模型路径").grid(row=0, column=0, sticky="w", pady=6)
        tk.Entry(frame, textvariable=self.model_path, width=72).grid(row=0, column=1, sticky="ew", pady=6)
        tk.Button(frame, text="选择模型", command=self.choose_model).grid(row=0, column=2, padx=8, pady=6)

        tk.Label(frame, text="输入类型").grid(row=1, column=0, sticky="w", pady=6)
        type_frame = tk.Frame(frame)
        type_frame.grid(row=1, column=1, sticky="w", pady=6)
        tk.Radiobutton(type_frame, text="单张图片", variable=self.source_type, value="image").pack(side="left")
        tk.Radiobutton(type_frame, text="文件夹", variable=self.source_type, value="folder").pack(side="left", padx=12)

        tk.Label(frame, text="输入路径").grid(row=2, column=0, sticky="w", pady=6)
        tk.Entry(frame, textvariable=self.source_path, width=72).grid(row=2, column=1, sticky="ew", pady=6)
        tk.Button(frame, text="打开路径", command=self.choose_source).grid(row=2, column=2, padx=8, pady=6)

        tk.Label(frame, text="置信度阈值").grid(row=3, column=0, sticky="w", pady=6)
        tk.Entry(frame, textvariable=self.conf_value, width=12).grid(row=3, column=1, sticky="w", pady=6)

        tk.Button(frame, text="开始检测", command=self.run_detection, width=18).grid(row=4, column=1, sticky="w", pady=14)

        tk.Label(frame, textvariable=self.status_text, anchor="w", justify="left", fg="blue").grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=10
        )

        frame.columnconfigure(1, weight=1)

    def choose_model(self):
        path = filedialog.askopenfilename(
            title="选择模型权重",
            filetypes=[("PyTorch Weights", "*.pt"), ("All Files", "*.*")],
            initialdir=str(ROOT),
        )
        if path:
            self.model_path.set(path)

    def choose_source(self):
        if self.source_type.get() == "image":
            path = filedialog.askopenfilename(
                title="选择图片",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.ppm"), ("All Files", "*.*")],
                initialdir=str(ROOT),
            )
        else:
            path = filedialog.askdirectory(title="选择图片文件夹", initialdir=str(ROOT))

        if path:
            self.source_path.set(path)

    def run_detection(self):
        model_path = Path(self.model_path.get()).expanduser()
        source_path = self.source_path.get().strip()

        if not model_path.exists():
            messagebox.showerror("错误", f"模型不存在:\n{model_path}")
            return

        if not source_path:
            messagebox.showerror("错误", "请先选择单张图片或文件夹。")
            return

        try:
            conf = float(self.conf_value.get())
        except ValueError:
            messagebox.showerror("错误", "置信度必须是数字，例如 0.25")
            return

        self.status_text.set("正在运行检测，请稍等...")
        self.root.update_idletasks()

        try:
            model = YOLO(str(model_path))
            results = model.predict(
                source=source_path, conf=conf, save=True,
                project=str(ROOT / "runs" / "detect"),
            )
        except Exception as exc:
            messagebox.showerror("检测失败", str(exc))
            self.status_text.set("检测失败，请检查路径、模型和环境。")
            return

        save_dir = None
        if results:
            save_dir = getattr(results[0], "save_dir", None)

        if save_dir:
            self.status_text.set(f"检测完成，结果已保存到:\n{save_dir}")
            messagebox.showinfo("完成", f"检测完成。\n结果保存到:\n{save_dir}")
        else:
            self.status_text.set("检测完成，但未获取到保存路径。")
            messagebox.showinfo("完成", "检测完成。")


def main():
    root = tk.Tk()
    DetectUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
