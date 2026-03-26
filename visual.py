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
        if not self.loss_data:
            return

        steps = []
        losses = []
        for entry in self.loss_data:
            if 'loss' in entry and 'step' in entry:
                steps.append(entry['step'])
                losses.append(entry['loss'])

        if not steps:
            return

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
        if not self.data:
            return

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
        if not self.data:
            return

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

    def plot_rouge_l(self):
        """Plot ROUGE-L scores comparison"""
        if not self.data:
            return

        # Check if rouge_l exists in the data
        if 'rouge_l' not in self.data.get('base', {}) or 'rouge_l' not in self.data.get('tuned', {}):
            print("ROUGE-L data not found, skipping plot")
            return

        plt.figure(figsize=(10, 6))
        models = ['Base Model', 'Fine-tuned']
        values = [self.data['base']['rouge_l'], self.data['tuned']['rouge_l']]

        colors = ['#f39c12', '#1abc9c']
        bars = plt.bar(models, values, color=colors, alpha=0.8, edgecolor='black')

        plt.title('ROUGE-L Score (Higher is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('ROUGE-L Score')
        plt.ylim(0, 1.0)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

        output_path = os.path.join(self.output_dir, 'rouge_l.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

    def plot_victorian_freq(self):
        """Plot Victorian term frequency comparison"""
        if not self.data:
            return

        # Check if victorian_freq exists in the data
        if 'victorian_freq' not in self.data.get('base', {}) or 'victorian_freq' not in self.data.get('tuned', {}):
            print("Victorian frequency data not found, skipping plot")
            return

        plt.figure(figsize=(10, 6))
        models = ['Base Model', 'Fine-tuned']
        values = [self.data['base']['victorian_freq'] * 100, self.data['tuned']['victorian_freq'] * 100]

        colors = ['#95a5a6', '#8e44ad']
        bars = plt.bar(models, values, color=colors, alpha=0.8, edgecolor='black')

        plt.title('Victorian/Holmesian Term Frequency (Higher is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('Percentage of Victorian Terms (%)')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

        output_path = os.path.join(self.output_dir, 'victorian_freq.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

    def plot_improvement(self):
        if not self.data:
            return

        plt.figure(figsize=(12, 6))

        base_ppl = self.data['base']['perplexity']
        tuned_ppl = self.data['tuned']['perplexity']
        ppl_imp = ((base_ppl - tuned_ppl) / base_ppl) * 100 if base_ppl > 0 else 0

        base_sim = self.data['base']['similarity']
        tuned_sim = self.data['tuned']['similarity']
        sim_imp = ((tuned_sim - base_sim) / base_sim) * 100 if base_sim > 0 else (100 if tuned_sim > 0 else 0)

        metrics = ['Perplexity\nReduction', 'Similarity\nIncrease']
        values = [ppl_imp, sim_imp]

        # Add ROUGE-L if available
        if 'rouge_l' in self.data.get('base', {}) and 'rouge_l' in self.data.get('tuned', {}):
            base_rouge = self.data['base']['rouge_l']
            tuned_rouge = self.data['tuned']['rouge_l']
            rouge_imp = ((tuned_rouge - base_rouge) / base_rouge) * 100 if base_rouge > 0 else 0
            metrics.append('ROUGE-L\nIncrease')
            values.append(rouge_imp)

        # Add Victorian freq if available
        if 'victorian_freq' in self.data.get('base', {}) and 'victorian_freq' in self.data.get('tuned', {}):
            base_vf = self.data['base']['victorian_freq']
            tuned_vf = self.data['tuned']['victorian_freq']
            vf_imp = ((tuned_vf - base_vf) / base_vf) * 100 if base_vf > 0 else 0
            metrics.append('Victorian\nTerms Increase')
            values.append(vf_imp)

        colors = ['#27ae60' if v > 0 else '#c0392b' for v in values]

        bars = plt.barh(metrics, values, color=colors, alpha=0.8, edgecolor='black')

        plt.title('Relative Model Improvement (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Percentage Change')
        plt.axvline(0, color='black', linewidth=0.8)

        for bar in bars:
            width = bar.get_width()
            label_x = width + (2 if width >= 0 else -2)
            plt.text(label_x, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', va='center',
                    ha='left' if width >= 0 else 'right', fontweight='bold')

        output_path = os.path.join(self.output_dir, 'improvement.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

    def create_dashboard(self):
        if not self.data:
            return

        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        # Perplexity comparison
        ax1 = fig.add_subplot(gs[0, 0])
        models = ['Base', 'Tuned']
        vals1 = [self.data['base']['perplexity'], self.data['tuned']['perplexity']]
        bars1 = ax1.bar(models, vals1, color=['#3498db', '#9b59b6'], alpha=0.8, edgecolor='black')
        ax1.set_title('Perplexity (Lower is Better)', fontweight='bold')
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')

        # Similarity comparison
        ax2 = fig.add_subplot(gs[0, 1])
        vals2 = [self.data['base']['similarity'], self.data['tuned']['similarity']]
        bars2 = ax2.bar(models, vals2, color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
        ax2.set_title('Jaccard Similarity (Higher is Better)', fontweight='bold')
        ax2.set_ylim(0, 1.0)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')

        # ROUGE-L comparison (if available)
        ax3 = fig.add_subplot(gs[1, 0])
        if 'rouge_l' in self.data.get('base', {}) and 'rouge_l' in self.data.get('tuned', {}):
            vals3 = [self.data['base']['rouge_l'], self.data['tuned']['rouge_l']]
            bars3 = ax3.bar(models, vals3, color=['#f39c12', '#1abc9c'], alpha=0.8, edgecolor='black')
            ax3.set_title('ROUGE-L Score (Higher is Better)', fontweight='bold')
            ax3.set_ylim(0, 1.0)
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'ROUGE-L Data Not Available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('ROUGE-L Score', fontweight='bold')

        # Victorian term frequency (if available)
        ax4 = fig.add_subplot(gs[1, 1])
        if 'victorian_freq' in self.data.get('base', {}) and 'victorian_freq' in self.data.get('tuned', {}):
            vals4 = [self.data['base']['victorian_freq'] * 100, self.data['tuned']['victorian_freq'] * 100]
            bars4 = ax4.bar(models, vals4, color=['#95a5a6', '#8e44ad'], alpha=0.8, edgecolor='black')
            ax4.set_title('Victorian Term Frequency (%)', fontweight='bold')
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}%', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'Victorian Freq Data Not Available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Victorian Term Frequency', fontweight='bold')

        # Improvement metrics
        ax5 = fig.add_subplot(gs[2, 0])
        base_ppl = self.data['base']['perplexity']
        tuned_ppl = self.data['tuned']['perplexity']
        ppl_imp = ((base_ppl - tuned_ppl) / base_ppl) * 100 if base_ppl > 0 else 0

        base_sim = self.data['base']['similarity']
        tuned_sim = self.data['tuned']['similarity']
        sim_imp = ((tuned_sim - base_sim) / base_sim) * 100 if base_sim > 0 else 0

        metrics = ['Perplexity Reduction', 'Similarity Increase']
        values = [ppl_imp, sim_imp]
        colors = ['#27ae60' if v > 0 else '#c0392b' for v in values]
        bars5 = ax5.barh(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        ax5.set_title('Percentage Improvement', fontweight='bold')
        ax5.axvline(0, color='black')
        for bar in bars5:
            width = bar.get_width()
            ax5.text(width, bar.get_y() + bar.get_height()/2, f' {width:.1f}%', va='center', fontweight='bold')

        # Training loss curve
        ax6 = fig.add_subplot(gs[2, 1])
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

                ax6.plot(steps, losses, marker='o', linestyle='-', color='#e74c3c', linewidth=2)
                ax6.set_title('Training Loss', fontweight='bold')
                ax6.set_xlabel('Steps')
                ax6.set_ylabel('Loss')
                ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No Loss Data Available', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Training Loss', fontweight='bold')

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
            self.plot_rouge_l()
            self.plot_victorian_freq()
            self.plot_improvement()
            self.plot_loss()
            self.create_dashboard()
            print("Visualization generation complete!")
        else:
            print("Skipping visualization due to missing data.")


if __name__ == "__main__":
    viz = ResultsVisualizer()
    viz.run()