import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def create_phase_cliff():
    """Create the Phase Cliff visualization showing structural break."""
    df = pd.read_csv('data/master_dataset.csv')
    
    # Sort by entropy for smooth line
    df_sorted = df.sort_values('instruction_entropy')
    
    # Define the kink threshold
    kink = 1000
    
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df['instruction_entropy'],
        y=df['ai_applicability_score'],
        mode='markers',
        name='Tasks',
        marker=dict(
            size=15,
            color=df['derived_wage'].fillna(0),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Market Wage (USD/hr)")
        ),
        text=df['Task ID'],
        hovertemplate='%{text}<br>Entropy: %{x:.0f}<br>AI Applicability: %{y:.2f}<extra></extra>'
    ))
    
    # Add the "Zone of Agentic Failure" shaded region
    fig.add_vrect(
        x0=kink, x1=df['instruction_entropy'].max(),
        fillcolor="red", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="<b>Zone of Agentic Failure</b>",
        annotation_position="top left"
    )
    
    # Add kink line
    fig.add_vline(
        x=kink,
        line_dash="dash",
        line_color="red",
        annotation_text="Complexity Threshold",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="<b>The Phase Cliff: Where AI Agents Become Liabilities</b>",
        xaxis_title="Instruction Entropy ($E$)",
        yaxis_title="AI Success Probability ($P$)",
        xaxis_type="log",
        template="plotly_white",
        height=600,
        font=dict(size=12)
    )
    
    fig.write_html('output/viz_1_phase_cliff.html')
    print("Phase Cliff visualization created: output/viz_1_phase_cliff.html")

def create_skill_radar():
    """Create the Skill Radar showing Human Moat."""
    
    # Define skill dimensions based on RLI task categories
    skills = ['Contextual Synthesis', 'Multi-Step Logic', 'Artifact Coupling', 
              'Domain Transfer', 'Constraint Satisfaction']
    
    # AI capabilities (normalized 0-1)
    ai_scores = [0.3, 0.4, 0.2, 0.35, 0.5]
    
    # Human capabilities (normalized 0-1)
    human_scores = [0.85, 0.90, 0.95, 0.80, 0.75]
    
    fig = go.Figure()
    
    # AI profile
    fig.add_trace(go.Scatterpolar(
        r=ai_scores,
        theta=skills,
        fill='toself',
        name='AI Agent',
        line_color='red',
        fillcolor='rgba(255, 0, 0, 0.3)'
    ))
    
    # Human profile
    fig.add_trace(go.Scatterpolar(
        r=human_scores,
        theta=skills,
        fill='toself',
        name='Human Expert',
        line_color='blue',
        fillcolor='rgba(0, 0, 255, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="<b>The Human Moat: Where Expertise Commands Premium</b>",
        template="plotly_white",
        height=600
    )
    
    fig.write_html('output/viz_4_skill_radar.html')
    print("Skill Radar visualization created: output/viz_4_skill_radar.html")

def create_agentic_shift():
    """Create Sankey diagram showing labor flow."""
    
    # Define flow from complexity levels to labor outcomes
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = ["Low Complexity Tasks", "Medium Complexity Tasks", "High Complexity Tasks",
                    "AI Automation", "Human Assisted", "Human Orchestrators"],
            color = ["lightblue", "yellow", "red", "green", "orange", "purple"]
        ),
        link = dict(
            source = [0, 0, 1, 1, 2, 2],
            target = [3, 4, 3, 4, 4, 5],
            value = [80, 20, 40, 60, 10, 90],
            color = ["rgba(0, 255, 0, 0.4)", "rgba(255, 165, 0, 0.4)", 
                    "rgba(0, 255, 0, 0.4)", "rgba(255, 165, 0, 0.4)",
                    "rgba(255, 165, 0, 0.4)", "rgba(128, 0, 128, 0.4)"]
        )
    )])

    fig.update_layout(
        title="<b>The Agentic Shift: Labor Flow in the 2026 Market</b>",
        font=dict(size=12),
        height=600
    )
    
    fig.write_html('output/viz_5_agentic_shift.html')
    print("Agentic Shift visualization created: output/viz_5_agentic_shift.html")

if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    
    create_phase_cliff()
    create_skill_radar()
    create_agentic_shift()
