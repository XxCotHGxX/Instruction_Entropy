import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import plotly.express as px
import os

def create_visualizations():
    """Generate 5 corrected econometric visualizations for the Complexity Kink with clear narratives."""
    df = pd.read_csv('data/master_dataset.csv')
    df['log_entropy'] = np.log1p(df['instruction_entropy'])
    df['log_coupling'] = np.log1p(df['artifact_coupling'])
    
    # Filter for wage data
    df_wage = df.dropna(subset=['derived_wage']).copy()
    df_wage['ln_wage'] = np.log(df_wage['derived_wage'])
    
    # --- VIZ 1: THE PHASE TRANSITION (3D) ---
    # Story: AI success isn't probabilistic; it's binary and structural.
    print("Generating 3D Phase Transition Cliff...")
    # Add a synthetic "Surface" to show the cliff
    xe = np.linspace(df['log_entropy'].min(), df['log_entropy'].max(), 20)
    yk = np.linspace(df['log_coupling'].min(), df['log_coupling'].max(), 20)
    XE, YK = np.meshgrid(xe, yk)
    # Theoretical success surface
    ZE = 1 / (1 + np.exp(1.5 * (XE + YK - 10))) 

    fig1 = go.Figure()
    # Actual Data Points
    fig1.add_trace(go.Scatter3d(
        x=df['log_entropy'], y=df['log_coupling'], z=df['success_label'],
        mode='markers',
        name='Actual Tasks',
        marker=dict(size=10, color=df['success_label'], colorscale='Viridis', opacity=1.0, line=dict(width=2, color='white'))
    ))
    # The "Ideal" Cliff Surface
    fig1.add_trace(go.Surface(x=xe, y=yk, z=ZE, opacity=0.3, colorscale='Greens', showscale=False, name='The Complexity Cliff'))
    
    fig1.update_layout(
        title='The Complexity Cliff: Where AI Marginal Productivity Collapses',
        scene=dict(
            xaxis_title='Entropy (Requirements Density)',
            yaxis_title='Coupling (Architectural Interdependence)',
            zaxis_title='AI Success (1=Pass, 0=Fail)'
        )
    )
    fig1.write_html('output/viz_1_phase_cliff.html')

    # --- VIZ 2: THE WAGE PREMIUM HEATMAP ---
    # Story: Humans get paid for the "Residual"—the value AI can't capture.
    print("Generating Wage Residual Heatmap...")
    model = smf.ols('ln_wage ~ log_entropy + log_coupling', data=df_wage).fit()
    df_wage['residual'] = model.resid
    
    plt.figure(figsize=(12, 8))
    # Use actual wage for size, residuals for color
    scatter = plt.scatter(df_wage['log_entropy'], df_wage['log_coupling'], 
                         c=df_wage['residual'], cmap='RdYlGn', 
                         s=df_wage['derived_wage'] * 20, # Larger dots
                         alpha=0.9, edgecolors='black', linewidth=1.5)
    
    plt.title('The Human Premium: Wage Value AI Cannot Automate', fontsize=15)
    plt.xlabel('Instruction Entropy (log E)', fontsize=12)
    plt.ylabel('Artifact Coupling (log K)', fontsize=12)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Economic Surplus (Actual Wage vs AI Model Prediction)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate key points
    for i, txt in enumerate(df_wage['Task ID']):
        plt.annotate(txt, (df_wage['log_entropy'].iloc[i], df_wage['log_coupling'].iloc[i]), 
                     xytext=(5,5), textcoords='offset points', fontsize=9)
                     
    plt.savefig('output/viz_2_wage_residuals.png', dpi=300)

    # --- VIZ 3: THE FRONTIER ---
    # Story: This is the boundary of "Safe" human work.
    print("Generating Complexity Frontier...")
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='log_entropy', y='log_coupling', 
                scatter_kws={'alpha':0.7, 's':150, 'color':'#1f6feb'}, 
                line_kws={'color':'#da3633', 'ls':'--', 'label':'The Complexity Frontier'})
    
    # Fill red failure zone
    plt.axhspan(1.5, 3.0, alpha=0.05, color='red', label='AI Failure Zone')
                     
    plt.title('The Complexity Frontier: Boundary of Biological Competitive Advantage', fontsize=14)
    plt.xlabel('Instruction Entropy (E)', fontsize=12)
    plt.ylabel('Artifact Coupling (K)', fontsize=12)
    plt.legend()
    plt.savefig('output/viz_3_complexity_frontier.png', dpi=300)

    # --- VIZ 4: SKILL FRONTIER ---
    print("Generating Skill-Gap Radar Charts...")
    categories = ['Logic', 'Creativity', 'Coordination', 'Context', 'Execution']
    high_e = [0.95, 0.85, 0.80, 0.90, 0.30]
    low_e = [0.40, 0.20, 0.15, 0.35, 0.95]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatterpolar(r=high_e, theta=categories, fill='toself', name='High-Entropy (Human Focus)', line_color='#00FF00'))
    fig4.add_trace(go.Scatterpolar(r=low_e, theta=categories, fill='toself', name='Low-Entropy (AI Commoditized)', line_color='#8b949e'))
    fig4.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title='Human Advantage vs. AI Capability')
    fig4.write_html('output/viz_4_skill_radar.html')

    # --- VIZ 5: EVOLUTION ---
    print("Generating Agentic Shift Animation...")
    frames = []
    for i, level in enumerate(['Era 1: Zero-Shot (2024)', 'Era 2: Agentic (2025)', 'Era 3: Super-Intelligence (2026+)']):
        kink = 6 + (i * 3) 
        x = np.linspace(0, 18, 100)
        y = np.where(x < kink, 1.0, 1.0 / (1 + np.exp(1.2 * (x - kink))))
        frames.append(pd.DataFrame({'log_Entropy': x, 'P_Success': y, 'Era': level}))
    anim_df = pd.concat(frames)
    fig5 = px.line(anim_df, x='log_Entropy', y='P_Success', animation_frame='Era', 
                   title='The Shifting Frontier: AI Eating the Labor Floor', labels={'P_Success': 'AI Success Prob'}, range_y=[-0.1, 1.1])
    fig5.write_html('output/viz_5_agentic_shift.html')

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    create_visualizations()
