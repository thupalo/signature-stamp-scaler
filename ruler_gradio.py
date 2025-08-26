import os
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import Tuple

import gradio as gr
from PIL import Image, ImageDraw
import numpy as np

from ruler_01 import process


def create_checkerboard_background(width: int, height: int, square_size: int = 20) -> Image.Image:
    """
    Create a light checkerboard pattern for transparency visualization.
    
    Args:
        width: Image width
        height: Image height
        square_size: Size of each checkerboard square in pixels
    
    Returns:
        PIL Image with checkerboard pattern
    """
    # Create checkerboard pattern
    checkerboard = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(checkerboard)
    
    # Light gray and white colors for subtle checkerboard
    color1 = (240, 240, 240)  # Very light gray
    color2 = (255, 255, 255)  # White
    
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            # Alternate colors in checkerboard pattern
            if (x // square_size + y // square_size) % 2 == 0:
                color = color1
            else:
                color = color2
            
            draw.rectangle([x, y, x + square_size, y + square_size], fill=color)
    
    return checkerboard


def add_checkerboard_background(image_path: str) -> str:
    """
    Add checkerboard background to an image to visualize transparency.
    
    Args:
        image_path: Path to the input image
    
    Returns:
        Path to the new image with checkerboard background
    """
    if not image_path or not os.path.exists(image_path):
        return image_path
    
    try:
        # Open the image
        img = Image.open(image_path).convert('RGBA')
        
        # Create checkerboard background
        checkerboard = create_checkerboard_background(img.width, img.height)
        
        # Composite the image over the checkerboard
        result = Image.alpha_composite(checkerboard.convert('RGBA'), img)
        
        # Save to a new temporary file
        temp_dir = Path(tempfile.gettempdir())
        output_path = temp_dir / f"checkerboard_{Path(image_path).name}"
        result.save(output_path, 'PNG')
        
        return str(output_path)
    
    except Exception as e:
        print(f"Error adding checkerboard background: {e}")
        return image_path


def run_conversion(input_path: str, debug: bool) -> Tuple[dict, str, str]:
    """
    input_path: path to input image file (local filepath)
    Returns: (debug_info dict, output_path, src_path)
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="parafka_gr_"))
    try:
        if not input_path:
            raise ValueError('No input provided')
        src = Path(input_path)
        if not src.exists():
            raise FileNotFoundError(f"Input file not found: {src}")

        out_path = tmpdir / 'output.png'

        # run main processing
        # process will overwrite the output path; ensure debug mode propagation
        from importlib import reload
        import ruler_01
        ruler_01.DEBUG = debug
        reload(ruler_01)

        process(str(src), str(out_path))

        debug_messages = {}
        if debug:
            # collect any debug images or messages
            dbg1 = Path('debug_info_1.png')
            dbg2 = Path('debug_info_2.png')
            debug_messages['debug_info_1'] = str(dbg1) if dbg1.exists() else ''
            debug_messages['debug_info_2'] = str(dbg2) if dbg2.exists() else ''

        return debug_messages, str(out_path), str(src)
    finally:
        # do not remove tmpdir immediately to allow Gradio to read the image; cleanup on server shutdown
        pass


# Prepare example files (point to workspace image if present)
EXAMPLES_DIR = Path(__file__).parent / 'examples'
EXAMPLES_DIR.mkdir(exist_ok=True)
workspace_example = Path(__file__).parent / 'image.jpg'
if workspace_example.exists():
    shutil.copy(str(workspace_example), str(EXAMPLES_DIR / 'image.jpg'))

EXAMPLES = [f.name for f in EXAMPLES_DIR.glob('*')]


def get_file_info(path: str) -> Tuple[str, str]:
    """Return (name, "<width> x <height> pixels, <bytes> bytes") when possible.
    For non-image or non-existing files, returns (name, "<bytes> bytes") or (name, '').
    """
    try:
        # If it's a URL, don't attempt to open here
        if isinstance(path, str) and (path.startswith('http://') or path.startswith('https://')):
            name = Path(path).name
            return name, ''

        p = Path(path)
        name = p.name
        def fmt_size(n: int) -> str:
            # human friendly size: bytes, kB or MB with 2 decimal places
            try:
                if n < 1024:
                    return f"{n} bytes"
                kb = n / 1024.0
                if kb < 1024:
                    return f"{kb:.2f} kB"
                mb = kb / 1024.0
                return f"{mb:.2f} MB"
            except Exception:
                return f"{n} bytes"

        if p.exists() and p.is_file():
            size_bytes = p.stat().st_size
            try:
                from PIL import Image
                with Image.open(p) as im:
                    w, h = im.size
                return name, f"{w} x {h} pixels, {fmt_size(size_bytes)}"
            except Exception:
                return name, fmt_size(size_bytes)
        else:
            return name, ''
    except Exception:
        return '', ''


PAGE_CSS = """
/* limit page width and center the app */
.gradio-container, .gradio-blocks, .container {
    max-width: 1024px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
"""

with gr.Blocks(title="Facsimile image generator", css=PAGE_CSS) as demo:
    gr.Markdown("# Facsimile image generator")
    gr.Markdown("Converting a photo containing a stamp, handwritten signature, and a ruler into a facsimile image, scaled to 300dpi with a transparent background.</br>" \
    "Drop an image or choose an example, then click Submit to convert.")

    with gr.Row():
        with gr.Column():
            gr.Markdown('### Input')
            # single input area: supports drop/browse and will also receive example selection
            # constrain displayed preview to 400x400 and scale the image to fit (no scrollbars)
            image_input = gr.Image(type='filepath', label='Drop image here or click to upload', interactive=True, width=400, height=400)
            # image_input serves as both the drop/browse control and the preview
            input_preview = image_input
            input_info = gr.Textbox(label='Info', interactive=False)

            submit = gr.Button('Submit')
            debug_checkbox = gr.Checkbox(label='Debug mode', value=False)

            # Prepare example file paths
            gallery_items = [str(EXAMPLES_DIR.joinpath(e)) for e in EXAMPLES[:3]]
            
            # Examples section - horizontal row of clickable preview images
            with gr.Row():
                gr.Markdown("**Examples:**")
            
            # Create example buttons in a horizontal row
            example_buttons = []
            if gallery_items:
                with gr.Row():
                    for i, item_path in enumerate(gallery_items[:3]):  # Show max 3 examples
                        try:
                            # Create a clickable image button for each example
                            btn = gr.Image(
                                value=item_path, 
                                label=f"Example {i+1}",
                                width=120,
                                height=120,
                                interactive=False,
                                show_label=False,
                                container=True
                            )
                            example_buttons.append((btn, item_path))
                        except Exception:
                            continue

            def on_example_click(path):
                """Handle clicking on an example image"""
                try:
                    if not path:
                        return gr.update(value=None), '', gr.update(visible=False), gr.update(value=None), '', gr.update(visible=False)
                    
                    name, info = get_file_info(path)
                    # Set the example as input and reset output
                    return gr.update(value=path), info, gr.update(visible=False), gr.update(value=None), '', gr.update(visible=False)
                except Exception:
                    return gr.update(value=None), '', gr.update(visible=False), gr.update(value=None), '', gr.update(visible=False)

            def on_gallery_select(*selected):
                # This function is no longer used - replaced with individual example button handlers
                pass

            # update input_info whenever the image_input value changes (drop or browse)
            def on_image_change(image_path):
                try:
                    if not image_path:
                        # clear info, hide download instruction, and reset output when input cleared
                        return '', gr.update(visible=False), gr.update(value=None), '', gr.update(visible=False)
                    _, info = get_file_info(image_path)
                    # when input changes, hide download instruction and reset output
                    return info, gr.update(visible=False), gr.update(value=None), '', gr.update(visible=False)
                except Exception:
                    return '', gr.update(visible=False), gr.update(value=None), '', gr.update(visible=False)

            # image_input.change will be wired later after download_button is declared

        with gr.Column():
            gr.Markdown('### Output')
            # constrain output preview to 400x400 and scale to fit (no scrollbars)
            # Disable built-in download since we want to download the original without checkerboard
            output_preview = gr.Image(label='Output (with transparency visualization)', width=400, height=400, show_download_button=False)
            output_info = gr.Textbox(label='Info', interactive=False)
            
            # Custom download for original file without checkerboard
            download_file = gr.File(label="Download Original PNG", visible=False)
            download_instruction = gr.Markdown("", visible=False)

            # state to hold the last generated output path
            last_output = gr.State('')

            # wire example buttons to set input when clicked
            for btn, path in example_buttons:
                btn.select(fn=lambda p=path: on_example_click(p), outputs=[input_preview, input_info, download_instruction, output_preview, output_info, download_file])
            
            # wire input change handler
            image_input.change(fn=on_image_change, inputs=[image_input], outputs=[input_info, download_instruction, output_preview, output_info, download_file])

            gr.Markdown('### Debug')
            debug_area = gr.Textbox(label='Debug messages', interactive=False, lines=8)


    def on_submit(image_path, debug_flag):
        # run conversion for the provided image path
        try:
            debug_msgs, out_path, src_path = run_conversion(image_path, debug_flag)
            # load input info
            # use the actual source file path returned by run_conversion for accurate info
            in_name, in_info = get_file_info(src_path)
            out_name, out_info = get_file_info(out_path)

            dbg_text = ''
            if debug_flag:
                for k, v in debug_msgs.items():
                    if v:
                        dbg_text += f"{k}: {v}\n"
            
            # Create version with checkerboard background for preview
            preview_path = add_checkerboard_background(out_path)
            
            # Update download instruction text
            download_text = "**<span style='color: green;'>✓ Processing complete!</span>** Download the original PNG file below (without checkerboard background):"
            
            return (gr.update(value=preview_path), 
                   out_info, 
                   dbg_text, 
                   gr.update(value=download_text, visible=True),
                   gr.update(value=out_path, visible=True))
        except Exception as e:
            # match outputs and hide download instruction on error
            return None, '', f'Error: {e}', gr.update(visible=False), gr.update(visible=False)
    
    # submit updates the output preview, output info, debug area, download instruction, and download file
    submit.click(fn=on_submit, inputs=[image_input, debug_checkbox], 
                outputs=[output_preview, output_info, debug_area, download_instruction, download_file])

    # Removed custom download button - using built-in Image component download functionality


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch Gradio interface for signature stamp scaler")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host to bind to (default: 0.0.0.0 for all interfaces)")
    parser.add_argument("--port", type=int, default=7860, 
                       help="Port to bind to (default: 7860)")
    parser.add_argument("--share", action="store_true", 
                       help="Create a public Gradio share link")
    parser.add_argument("--local", action="store_true", 
                       help="Run in local mode only (127.0.0.1)")
    
    args = parser.parse_args()
    
    # Determine host based on arguments
    host = "127.0.0.1" if args.local else args.host
    
    print(f"Starting Gradio interface...")
    print(f"Local URL: http://127.0.0.1:{args.port}")
    if not args.local:
        print(f"Network URL: http://{host}:{args.port}")
        print("Note: Make sure your firewall allows connections on this port")
    
    # Launch with network access enabled
    demo.launch(
        server_name=host,       # Listen on specified host
        server_port=args.port,  # Port number
        share=args.share        # Public share link if requested
    )
