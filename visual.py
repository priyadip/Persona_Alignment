import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

class ResultsVisualizer:
    def __init__(self, evaluation_file="evaluation_report.json", loss_file="training_loss.json", output_dir="result"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            with open(evaluation_file, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {evaluation_file} not found.")
            self.data = None

        try:
            with open(loss_file, 'r') as f:
                self.loss_data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {loss_file} not found.")
            self.loss_data = None

        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12})

    def plot_loss(self):
        if not self.loss_data: return
        
        steps = []
        losses = []
        for entry in self.loss_data:
            if 'loss' in entry and 'step' in entry:
                steps.append(entry['step'])
                losses.append(entry['loss'])
        
        if not steps: return

        if len(steps) > 50:
            skip = len(steps) // 50
            steps = steps[::skip]
            losses = losses[::skip]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, marker='o', linestyle='-', color='#e74c3c', linewidth=2)
        
        plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.output_dir, 'training_loss.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

    def plot_perplexity(self):
        if not self.data: return
        
        plt.figure(figsize=(10, 6))
        models = ['Base Model', 'Fine-tuned']
        values = [self.data['base']['perplexity'], self.data['tuned']['perplexity']]
        
        colors = ['#3498db', '#9b59b6']
        bars = plt.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
        
        plt.title('Perplexity Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('Perplexity Score')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
            
        output_path = os.path.join(self.output_dir, 'perplexity.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

    def plot_similarity(self):
        if not self.data: return

        plt.figure(figsize=(10, 6))
        models = ['Base Model', 'Fine-tuned']
        values = [self.data['base']['similarity'], self.data['tuned']['similarity']]
        
        colors = ['#e74c3c', '#2ecc71']
        bars = plt.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
        
        plt.title('Content Similarity (Higher is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('Jaccard Similarity Score')
        plt.ylim(0, 1.0) 

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
            
        output_path = os.path.join(self.output_dir, 'similarity.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

    def plot_improvement(self):
        if not self.data: return

        plt.figure(figsize=(10, 6))

        base_ppl = self.data['base']['perplexity']
        tuned_ppl = self.data['tuned']['perplexity']

        ppl_imp = ((base_ppl - tuned_ppl) / base_ppl) * 100
        
        base_sim = self.data['base']['similarity']
        tuned_sim = self.data['tuned']['similarity']

        if base_sim > 0:
            sim_imp = ((tuned_sim - base_sim) / base_sim) * 100
        else:
            sim_imp = 100 if tuned_sim > 0 else 0
        
        metrics = ['Perplexity\nReduction', 'Similarity\nIncrease']
        values = [ppl_imp, sim_imp]
        colors = ['#27ae60' if v > 0 else '#c0392b' for v in values]
        
        bars = plt.barh(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        
        plt.title('Relative Model Improvement (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Percentage Change')
        plt.axvline(0, color='black', linewidth=0.8)

        for bar in bars:
            width = bar.get_width()
            label_x = width + (1 if width >= 0 else -1)
            plt.text(label_x, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', va='center', 
                    ha='left' if width >= 0 else 'right', fontweight='bold')
            
        output_path = os.path.join(self.output_dir, 'improvement.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

    def create_dashboard(self):
        if not self.data: return

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        models = ['Base', 'Tuned']
        vals1 = [self.data['base']['perplexity'], self.data['tuned']['perplexity']]
        bars1 = ax1.bar(models, vals1, color=['#3498db', '#9b59b6'], alpha=0.8, edgecolor='black')
        ax1.set_title('Perplexity (Lower is Better)', fontweight='bold')
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')

        ax2 = fig.add_subplot(gs[0, 1])
        vals2 = [self.data['base']['similarity'], self.data['tuned']['similarity']]
        bars2 = ax2.bar(models, vals2, color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
        ax2.set_title('Similarity (Higher is Better)', fontweight='bold')
        ax2.set_ylim(0, 1.0)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')

        ax3 = fig.add_subplot(gs[1, 0])
        base_ppl = self.data['base']['perplexity']
        tuned_ppl = self.data['tuned']['perplexity']
        ppl_imp = ((base_ppl - tuned_ppl) / base_ppl) * 100
        
        base_sim = self.data['base']['similarity']
        tuned_sim = self.data['tuned']['similarity']
        sim_imp = ((tuned_sim - base_sim) / base_sim) * 100 if base_sim > 0 else 0
        
        metrics = ['Perplexity Reduction', 'Similarity Increase']
        values = [ppl_imp, sim_imp]
        colors = ['#27ae60' if v > 0 else '#c0392b' for v in values]
        bars3 = ax3.barh(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_title('Percentage Improvement', fontweight='bold')
        ax3.axvline(0, color='black')
        for bar in bars3:
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2, f' {width:.1f}%', va='center', fontweight='bold')

        ax4 = fig.add_subplot(gs[1, 1])
        if self.loss_data:
            steps = []
            losses = []
            for entry in self.loss_data:
                if 'loss' in entry and 'step' in entry:
                    steps.append(entry['step'])
                    losses.append(entry['loss'])
            if steps:
                if len(steps) > 50:
                    skip = len(steps) // 50
                    steps = steps[::skip]
                    losses = losses[::skip]
                
                ax4.plot(steps, losses, marker='o', linestyle='-', color='#e74c3c', linewidth=2)
                ax4.set_title('Training Loss', fontweight='bold')
                ax4.set_xlabel('Steps')
                ax4.set_ylabel('Loss')
                ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Loss Data Available', ha='center', va='center')

        plt.suptitle('Sherlock Model Evaluation Dashboard', fontsize=20, fontweight='bold')
        
        output_path = os.path.join(self.output_dir, 'dashboard.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

    def run(self):
        print(f"Generating visualizations in '{self.output_dir}'...")
        if self.data:
            self.plot_perplexity()
            self.plot_similarity()
            self.plot_improvement()
            self.plot_loss()
            self.create_dashboard()
            print("Visualization generation complete!")
        else:
            print("Skipping visualization due to missing data.")

if __name__ == "__main__":
    viz = ResultsVisualizer()
    viz.run()