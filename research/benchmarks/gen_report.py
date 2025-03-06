import pandas as pd
import os

def generate_html_report(csv_file, output_html):
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Select required columns
    df = df[['image_name', 'user_prompt', 'overlay_path', 'is_nsfw', 'detected_nsfw']]
    
    # Generate HTML
    html_content = """
    <html>
    <head>
        <title>Image Report</title>
        <style>
            table { width: 100%; border-collapse: collapse; }
            th, td { border: 1px solid black; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            img { max-width: 150px; max-height: 150px; }
        </style>
    </head>
    <body>
        <h2>Image Report</h2>
        <table>
            <tr>
                <th>Image Name</th>
                <th>User Prompt</th>
                <th>Image Overlay</th>
                <th>Is NSFW</th>
                <th>Detected NSFW</th>
            </tr>
    """
    
    for _, row in df.iterrows():
        image_overlay_path = row['overlay_path'] if os.path.exists(row['overlay_path']) else 'not_found.png'
        html_content += f"""
            <tr>
                <td>{row['image_name']}</td>
                <td>{row['user_prompt']}</td>
                <td><img src='{image_overlay_path}' alt='Overlay Image'></td>
                <td>{row['is_nsfw']}</td>
                <td>{row['detected_nsfw']}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Write to HTML file
    with open(output_html, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    print(f"Report generated: {output_html}")

# Example usage
generate_html_report('output.csv', 'report.html')
