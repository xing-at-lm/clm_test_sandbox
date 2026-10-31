# -*- coding: utf-8 -*-
"""
Created on Mon May 25 21:20:48 2026

@author: XingPan
"""
import io
from pptx import Presentation
from pptx.util import Inches, Pt
import matplotlib.pyplot as plt
from pptx.enum.text import PP_ALIGN
import pandas as pd

class PresentationManager:
    def __init__(self, template_path=None):
        """
        Initialize with an optional template.
        If template_path is provided, the new pptx will inherit all 
        styles, masters, and layouts from that file.
        """
        self.prs = Presentation(template_path) if template_path else Presentation()
        # Define some common layout indices (standard PPT layouts)
        self.TITLE_SLIDE = 0
        self.BULLET_SLIDE = 1
        self.BLANK_SLIDE = 6      
        
    def list_layouts(self):
        """Prints all available layout names in the current template."""
        print("Available Layouts:")
        for i, layout in enumerate(self.prs.slide_layouts):
            print(f"Index {i}: {layout.name}")

    def add_slide_by_layout_name(self, layout_name):
        """
        Helper to find a layout by its string name (e.g., 'Title Slide')
        instead of guessing the index.
        """
        selected_layout = None
        for layout in self.prs.slide_layouts:
            if layout.name == layout_name:
                selected_layout = layout
                break
        
        if not selected_layout:
            # Fallback to the first layout if name not found
            selected_layout = self.prs.slide_layouts[0]
            print(f"Warning: Layout '{layout_name}' not found. Using default.")
            
        return self.prs.slides.add_slide(selected_layout)

    def add_title_slide(self, title_text, subtitle_text, layout_name='Title Slide'):
        slide = self.add_slide_by_layout_name(layout_name)
        # Standard templates usually have Title at 0 and Subtitle at 1
        slide.shapes.title.text = title_text
        if len(slide.placeholders) > 1:
            slide.placeholders[1].text = subtitle_text

    def add_table_slide(self, title_text, df, layout_name='Title and Content'):
        slide = self.add_slide_by_layout_name(layout_name)
        slide.shapes.title.text = title_text

        rows, cols = df.shape
        # Create table (adjusting position to fit typical templates)
        left, top, width, height = Inches(0.5), Inches(1.5), Inches(9), Inches(0.8)
        table = slide.shapes.add_table(rows + 1, cols, left, top, width, height).table

        # Fill Header
        for col_idx, col_name in enumerate(df.columns):
            table.cell(0, col_idx).text = str(col_name)

        # Fill Data
        for row_idx, row in enumerate(df.values):
            for col_idx, value in enumerate(row):
                table.cell(row_idx + 1, col_idx).text = str(value)

    def add_plot_slide(self, title_text, fig, left=1, top=1.5, width=8):
        """
        Takes a Matplotlib figure object and inserts it into a new slide.
        """
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[self.BULLET_SLIDE])
        slide.shapes.title.text = title_text
        
        # Save the matplotlib figure to a 'buffer' (in-memory file)
        image_stream = io.BytesIO()
        fig.savefig(image_stream, format='png', bbox_inches='tight')
        image_stream.seek(0)
        
        # Add the image to the slide
        slide.shapes.add_picture(
            image_stream, 
            Inches(left), 
            Inches(top), 
            width=Inches(width)
        )

    def add_bullet_slide(self, title_text, points, font_size=None):
        """
        Creates a slide with a title and a list of bullet points.
        'points' should be a list of strings.
        'font_size' is an optional integer parameter to override the template's font size (in Pt).
        """
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[self.BULLET_SLIDE])
        slide.shapes.title.text = title_text
        
        tf = slide.placeholders[1].text_frame
        tf.clear()  # Clear default text
        
        for point in points:
            p = tf.add_paragraph()
            p.text = point
            p.level = 0  # 0 is top-level bullet
            
            # Apply the custom font size if one was passed in
            if font_size is not None:
                p.font.size = Pt(font_size)

    def _format_cell(self, cell, is_header=False):
        """Internal helper to style cells."""
        paragraph = cell.text_frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.CENTER
        run = paragraph.runs[0]
        run.font.size = Pt(14)
        if is_header:
            run.font.bold = True
            # Optional: You can set background colors here if using a template

    def save(self, filename="output.pptx"):
        self.prs.save(filename)
        print(f"Presentation saved as {filename}")

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize
    maker = PresentationManager()

    # 2. Add Title
    maker.add_title_slide("Quarterly Data Report", "Generated via Python Automation")

    # 3. Add Bullets (Now showcasing custom font size)
    maker.add_bullet_slide(
        title_text="Key Takeaways", 
        points=[
            "Revenue increased by 15% YoY.",
            "Operational costs remained flat.",
            "New user acquisition is the primary growth driver."
        ],
        font_size=28  # Explicitly setting font size to 28 Pt
    )

    # 4. Create a Matplotlib Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Q1', 'Q2', 'Q3', 'Q4'], [10, 25, 15, 30], color='skyblue')
    ax.set_title("Annual Sales Performance")
    
    # 5. Add Plot to Slide
    maker.add_plot_slide("Financial Performance Graph", fig)

    # Create dummy data
    data = {
        'Region': ['North', 'South', 'East', 'West'],
        'Sales (k)': [450, 620, 310, 580],
        'Growth': ['12%', '18%', '-5%', '22%'],
        'Status': ['On Track', 'Exceeded', 'Review', 'On Track']
    }
    df_sales = pd.DataFrame(data)

    # Add the table slide
    maker.add_table_slide("Regional Sales Data", df_sales)
    
    # 6. Save
    maker.save("My_Automation_Report.pptx")