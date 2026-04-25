import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_figure_1():
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Define box properties
    box_width = 16
    box_height = 60
    y_start = 20
    spacing = 20
    
    colors = ['#f8f9fa', '#e9ecef', '#dee2e6', '#ced4da', '#adb5bd']
    edge_color = '#343a40'
    
    # Define the content for each box
    steps = [
        {
            "title": "Cohort\nConstruction",
            "content": "ADNI tau90 / tau180\nOASIS3 tau90 / tau180\nNACC strict A/T"
        },
        {
            "title": "Feature\nHarmonization",
            "content": "Demo/genetic variables\nSeverity variables\nSubject-level tables"
        },
        {
            "title": "OASIS\nFeature Audit",
            "content": "Recover age and APOE\nConfirm MoCA and\nCDR global absent"
        },
        {
            "title": "Analyses",
            "content": "Severity-strip ablation\nMatched-feature rerun\nBootstrap CI (primary)"
        },
        {
            "title": "Outputs",
            "content": "Table 1\nPrimary comparison\nSupplementary analyses"
        }
    ]

    # Draw boxes and text
    for i, step in enumerate(steps):
        x_start = 2 + (i * spacing)
        
        # Draw the main box
        rect = patches.FancyBboxPatch(
            (x_start, y_start), box_width, box_height,
            boxstyle="round,pad=0.5", linewidth=1.5, edgecolor=edge_color, facecolor='white', zorder=1
        )
        ax.add_patch(rect)
        
        # Add Title (Bold)
        plt.text(x_start + box_width/2, y_start + box_height - 8, step["title"], 
                 ha='center', va='center', fontsize=12, fontweight='bold', family='sans-serif')
        
        # Add Content
        plt.text(x_start + box_width/2, y_start + box_height/2 - 5, step["content"], 
                 ha='center', va='center', fontsize=10, family='sans-serif', linespacing=1.8)

        # Draw connecting arrows
        if i < len(steps) - 1:
            ax.annotate('', xy=(x_start + box_width + 3.5, y_start + box_height/2), 
                        xytext=(x_start + box_width + 0.5, y_start + box_height/2),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color=edge_color))

    # Add Figure Title and Subtitle
    plt.text(50, 95, "Study workflow for cross-cohort tau-positivity harmonization analysis", 
             ha='center', va='center', fontsize=16, fontweight='bold', family='sans-serif')
    plt.text(50, 90, "From cohort assembly and harmonization to matched-feature comparison and publication outputs", 
             ha='center', va='center', fontsize=11, style='italic', family='sans-serif')

    plt.tight_layout()
    plt.savefig('figure1_workflow_schematic.pdf', dpi=300, bbox_inches='tight')
    print("Figure 1 saved as 'figure1_workflow_schematic.pdf'")
    plt.show()

if __name__ == "__main__":
    create_figure_1()