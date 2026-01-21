"""
Ultra-Advanced Performance Analytics and Monitoring Suite

This module provides comprehensive performance monitoring, analytics, and
diagnostic tools for Bayesian optimization systems and advanced algorithms.

Key Features:
- Real-time performance monitoring
- Advanced convergence analysis
- Resource utilization tracking
- Statistical performance metrics
- Comparative analysis tools
- Performance bottleneck detection
- Automated performance reports
- Scalability analysis
- Memory profiling
- Computational efficiency metrics
"""

import numpy as np
import pandas as pd
import time
import psutil
import threading
import queue
from collections import defaultdict, deque
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Optional, Dict, List, Tuple, Union, Callable, Any
from abc import ABC, abstractmethod
import json
import pickle
import os
from dataclasses import dataclass, field
from enum import Enum


class MetricType(Enum):
    """Enumeration of performance metric types."""
    TIMING = "timing"
    MEMORY = "memory"
    CPU = "cpu"
    CONVERGENCE = "convergence"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    SCALABILITY = "scalability"
    RESOURCE = "resource"


@dataclass
class PerformanceMetric:
    """Data class for performance metrics."""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePerformanceMonitor(ABC):
    """Abstract base class for performance monitors."""
    
    def __init__(self, sampling_interval=1.0, max_history=1000):
        """
        Initialize performance monitor.
        
        Parameters
        ----------
        sampling_interval : float, default=1.0
            Interval between samples in seconds
        max_history : int, default=1000
            Maximum number of samples to keep in history
        """
        self.sampling_interval = sampling_interval
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.is_monitoring = False
        self.monitor_thread = None
        
    @abstractmethod
    def collect_metrics(self) -> List[PerformanceMetric]:
        """Collect performance metrics."""
        pass
    
    def start_monitoring(self):
        """Start performance monitoring in background thread."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self.collect_metrics()
                for metric in metrics:
                    self.metrics_history.append(metric)
                time.sleep(self.sampling_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                
    def get_metrics(self, metric_type=None, time_window=None):
        """
        Get collected metrics.
        
        Parameters
        ----------
        metric_type : MetricType, optional
            Filter by metric type
        time_window : timedelta, optional
            Filter by time window
            
        Returns
        -------
        metrics : List[PerformanceMetric]
            Filtered metrics
        """
        metrics = list(self.metrics_history)
        
        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]
            
        if time_window:
            cutoff_time = datetime.now() - time_window
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
        return metrics
    
    def get_metrics_dataframe(self, **filters):
        """Get metrics as pandas DataFrame."""
        metrics = self.get_metrics(**filters)
        
        if not metrics:
            return pd.DataFrame()
            
        data = []
        for metric in metrics:
            data.append({
                'name': metric.name,
                'value': metric.value,
                'timestamp': metric.timestamp,
                'type': metric.metric_type.value,
                'unit': metric.unit,
                **metric.metadata
            })
            
        return pd.DataFrame(data)


class SystemResourceMonitor(BasePerformanceMonitor):
    """Monitor system resources (CPU, memory, etc.)."""
    
    def __init__(self, sampling_interval=1.0, max_history=1000):
        """Initialize system resource monitor."""
        super().__init__(sampling_interval, max_history)
        self.process = psutil.Process()
        
    def collect_metrics(self):
        """Collect system resource metrics."""
        metrics = []
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        metrics.append(PerformanceMetric(
            name="cpu_percent",
            value=cpu_percent,
            timestamp=timestamp,
            metric_type=MetricType.CPU,
            unit="%"
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(PerformanceMetric(
            name="memory_percent",
            value=memory.percent,
            timestamp=timestamp,
            metric_type=MetricType.MEMORY,
            unit="%"
        ))
        
        metrics.append(PerformanceMetric(
            name="memory_available_gb",
            value=memory.available / (1024**3),
            timestamp=timestamp,
            metric_type=MetricType.MEMORY,
            unit="GB"
        ))
        
        # Process-specific metrics
        process_memory = self.process.memory_info()
        metrics.append(PerformanceMetric(
            name="process_memory_mb",
            value=process_memory.rss / (1024**2),
            timestamp=timestamp,
            metric_type=MetricType.MEMORY,
            unit="MB"
        ))
        
        process_cpu = self.process.cpu_percent()
        metrics.append(PerformanceMetric(
            name="process_cpu_percent",
            value=process_cpu,
            timestamp=timestamp,
            metric_type=MetricType.CPU,
            unit="%"
        ))
        
        return metrics


class OptimizationPerformanceMonitor(BasePerformanceMonitor):
    """Monitor optimization algorithm performance."""
    
    def __init__(self, sampling_interval=None, max_history=1000):
        """Initialize optimization performance monitor."""
        super().__init__(sampling_interval or 0.1, max_history)
        self.optimization_start_time = None
        self.iteration_count = 0
        self.best_values = []
        self.current_values = []
        self.evaluation_times = []
        
    def start_optimization(self):
        """Start optimization session."""
        self.optimization_start_time = time.time()
        self.iteration_count = 0
        self.best_values = []
        self.current_values = []
        self.evaluation_times = []
        
    def record_iteration(self, x, y, evaluation_time=None):
        """Record optimization iteration."""
        timestamp = datetime.now()
        self.iteration_count += 1
        
        if evaluation_time is None:
            evaluation_time = time.time() - (self.optimization_start_time or time.time())
            
        self.current_values.append(y)
        
        if not self.best_values or y < min(self.best_values):
            self.best_values.append(y)
        else:
            self.best_values.append(self.best_values[-1] if self.best_values else y)
            
        self.evaluation_times.append(evaluation_time)
        
        # Create metrics
        metrics = []
        
        # Convergence metrics
        metrics.append(PerformanceMetric(
            name="best_value",
            value=min(self.best_values),
            timestamp=timestamp,
            metric_type=MetricType.CONVERGENCE
        ))
        
        metrics.append(PerformanceMetric(
            name="current_value",
            value=y,
            timestamp=timestamp,
            metric_type=MetricType.CONVERGENCE
        ))
        
        # Improvement metrics
        if len(self.best_values) > 1:
            improvement = self.best_values[-2] - self.best_values[-1]
            metrics.append(PerformanceMetric(
                name="improvement",
                value=improvement,
                timestamp=timestamp,
                metric_type=MetricType.CONVERGENCE
            ))
            
        # Timing metrics
        metrics.append(PerformanceMetric(
            name="evaluation_time",
            value=evaluation_time,
            timestamp=timestamp,
            metric_type=MetricType.TIMING,
            unit="seconds"
        ))
        
        metrics.append(PerformanceMetric(
            name="cumulative_time",
            value=time.time() - (self.optimization_start_time or time.time()),
            timestamp=timestamp,
            metric_type=MetricType.TIMING,
            unit="seconds"
        ))
        
        # Efficiency metrics
        if self.iteration_count > 1:
            efficiency = (self.best_values[0] - min(self.best_values)) / sum(self.evaluation_times)
            metrics.append(PerformanceMetric(
                name="optimization_efficiency",
                value=efficiency,
                timestamp=timestamp,
                metric_type=MetricType.EFFICIENCY
            ))
            
        return metrics
    
    def collect_metrics(self):
        """Collect current optimization metrics."""
        if not self.optimization_start_time:
            return []
            
        timestamp = datetime.now()
        metrics = []
        
        # Current status metrics
        metrics.append(PerformanceMetric(
            name="iteration_count",
            value=self.iteration_count,
            timestamp=timestamp,
            metric_type=MetricType.CONVERGENCE
        ))
        
        if self.best_values:
            metrics.append(PerformanceMetric(
                name="convergence_rate",
                value=self._compute_convergence_rate(),
                timestamp=timestamp,
                metric_type=MetricType.CONVERGENCE
            ))
            
        return metrics
    
    def _compute_convergence_rate(self):
        """Compute current convergence rate."""
        if len(self.best_values) < 10:
            return 0.0
            
        # Compute convergence rate over last 10 iterations
        recent_values = self.best_values[-10:]
        if len(recent_values) < 2:
            return 0.0
            
        # Linear regression on log scale
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        try:
            slope, _, _, _, _ = stats.linregress(x, y)
            return abs(slope)
        except:
            return 0.0


class ConvergenceAnalyzer:
    """Advanced convergence analysis tools."""
    
    def __init__(self):
        """Initialize convergence analyzer."""
        self.convergence_history = []
        
    def analyze_convergence(self, values, window_size=10):
        """
        Analyze convergence of optimization values.
        
        Parameters
        ----------
        values : array-like
            Sequence of objective values
        window_size : int, default=10
            Size of analysis window
            
        Returns
        -------
        analysis : dict
            Convergence analysis results
        """
        values = np.asarray(values)
        
        if len(values) < 2:
            return {'status': 'insufficient_data'}
            
        analysis = {}
        
        # Basic statistics
        analysis['best_value'] = np.min(values)
        analysis['final_value'] = values[-1]
        analysis['improvement'] = values[0] - values[-1]
        analysis['relative_improvement'] = analysis['improvement'] / abs(values[0]) if values[0] != 0 else 0
        
        # Convergence detection
        analysis['converged'] = self._detect_convergence(values, window_size)
        analysis['convergence_iteration'] = self._find_convergence_point(values, window_size)
        
        # Convergence rate
        analysis['convergence_rate'] = self._compute_convergence_rate(values)
        
        # Plateau detection
        analysis['plateaus'] = self._detect_plateaus(values, window_size)
        
        # Stagnation detection
        analysis['stagnation_periods'] = self._detect_stagnation(values, window_size)
        
        return analysis
    
    def _detect_convergence(self, values, window_size):
        """Detect if sequence has converged."""
        if len(values) < window_size:
            return False
            
        # Check if recent improvements are small
        recent_values = values[-window_size:]
        improvements = np.diff(recent_values)
        
        # Converged if all recent improvements are small
        threshold = 1e-6 * abs(values[0]) if values[0] != 0 else 1e-6
        return np.all(np.abs(improvements) < threshold)
    
    def _find_convergence_point(self, values, window_size):
        """Find iteration where convergence occurred."""
        if len(values) < window_size:
            return None
            
        for i in range(window_size, len(values)):
            window_values = values[i-window_size:i]
            if self._detect_convergence(window_values, window_size):
                return i
                
        return None
    
    def _compute_convergence_rate(self, values):
        """Compute convergence rate."""
        if len(values) < 3:
            return 0.0
            
        # Fit exponential decay model
        try:
            x = np.arange(len(values))
            y = np.array(values)
            
            # Log transform for exponential fit
            if np.all(y > 0):
                log_y = np.log(y)
                slope, _, _, _, _ = stats.linregress(x, log_y)
                return -slope  # Negative slope indicates convergence
            else:
                # Linear fit for non-positive values
                slope, _, _, _, _ = stats.linregress(x, y)
                return abs(slope)
        except:
            return 0.0
    
    def _detect_plateaus(self, values, window_size, tolerance=1e-6):
        """Detect plateau regions."""
        plateaus = []
        
        if len(values) < window_size:
            return plateaus
            
        i = 0
        while i < len(values) - window_size:
            window = values[i:i+window_size]
            if np.std(window) < tolerance:
                plateaus.append((i, i+window_size))
                i += window_size
            else:
                i += 1
                
        return plateaus
    
    def _detect_stagnation(self, values, window_size, threshold=0.01):
        """Detect periods of stagnation."""
        stagnation_periods = []
        
        if len(values) < window_size:
            return stagnation_periods
            
        for i in range(window_size, len(values)):
            window_values = values[i-window_size:i]
            improvements = np.abs(np.diff(window_values))
            
            if np.mean(improvements) < threshold:
                stagnation_periods.append((i-window_size, i))
                
        return stagnation_periods


class ScalabilityAnalyzer:
    """Analyze scalability of optimization algorithms."""
    
    def __init__(self):
        """Initialize scalability analyzer."""
        self.scalability_data = {}
        
    def record_scalability_test(self, problem_size, n_iterations, 
                                execution_time, memory_usage, accuracy):
        """
        Record scalability test results.
        
        Parameters
        ----------
        problem_size : int
            Size of the problem (e.g., dimensions)
        n_iterations : int
            Number of iterations performed
        execution_time : float
            Total execution time
        memory_usage : float
            Peak memory usage
        accuracy : float
            Final accuracy achieved
        """
        if problem_size not in self.scalability_data:
            self.scalability_data[problem_size] = []
            
        self.scalability_data[problem_size].append({
            'n_iterations': n_iterations,
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'accuracy': accuracy,
            'time_per_iteration': execution_time / n_iterations,
            'memory_per_iteration': memory_usage / n_iterations
        })
        
    def analyze_scalability(self):
        """Analyze scalability across problem sizes."""
        if not self.scalability_data:
            return {}
            
        analysis = {}
        
        # Extract data
        problem_sizes = sorted(self.scalability_data.keys())
        avg_times = []
        avg_memory = []
        avg_accuracy = []
        
        for size in problem_sizes:
            data = self.scalability_data[size]
            avg_times.append(np.mean([d['execution_time'] for d in data]))
            avg_memory.append(np.mean([d['memory_usage'] for d in data]))
            avg_accuracy.append(np.mean([d['accuracy'] for d in data]))
            
        # Time complexity analysis
        if len(problem_sizes) >= 3:
            # Fit polynomial to log-log data
            log_sizes = np.log(problem_sizes)
            log_times = np.log(avg_times)
            
            slope, _, _, _, _ = stats.linregress(log_sizes, log_times)
            analysis['time_complexity_exponent'] = slope
            
            # Memory complexity analysis
            log_memory = np.log(avg_memory)
            slope_mem, _, _, _, _ = stats.linregress(log_sizes, log_memory)
            analysis['memory_complexity_exponent'] = slope_mem
            
        # Efficiency analysis
        analysis['efficiency_degradation'] = self._compute_efficiency_degradation(
            problem_sizes, avg_times
        )
        
        # Accuracy scaling
        analysis['accuracy_scaling'] = self._analyze_accuracy_scaling(
            problem_sizes, avg_accuracy
        )
        
        return analysis
    
    def _compute_efficiency_degradation(self, problem_sizes, execution_times):
        """Compute how efficiency degrades with problem size."""
        if len(problem_sizes) < 2:
            return 0.0
            
        # Compute time per unit problem size
        efficiency = [time / size for size, time in zip(problem_sizes, execution_times)]
        
        # Degradation is the increase in efficiency ratio
        if len(efficiency) >= 2:
            return efficiency[-1] / efficiency[0]
        else:
            return 1.0
    
    def _analyze_accuracy_scaling(self, problem_sizes, accuracies):
        """Analyze how accuracy scales with problem size."""
        if len(problem_sizes) < 2:
            return {}
            
        # Compute accuracy degradation
        accuracy_degradation = accuracies[0] - accuracies[-1]
        
        # Fit linear model
        slope, _, r_value, _, _ = stats.linregress(problem_sizes, accuracies)
        
        return {
            'degradation': accuracy_degradation,
            'slope': slope,
            'r_squared': r_value**2
        }


class PerformanceReporter:
    """Generate comprehensive performance reports."""
    
    def __init__(self):
        """Initialize performance reporter."""
        self.monitors = {}
        self.analyzers = {}
        
    def add_monitor(self, name, monitor):
        """Add a performance monitor."""
        self.monitors[name] = monitor
        
    def add_analyzer(self, name, analyzer):
        """Add a performance analyzer."""
        self.analyzers[name] = analyzer
        
    def generate_report(self, output_format='dict'):
        """
        Generate comprehensive performance report.
        
        Parameters
        ----------
        output_format : str, default='dict'
            Output format: 'dict', 'json', 'html'
            
        Returns
        -------
        report : dict or str
            Performance report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitors': {},
            'analyzers': {},
            'summary': {}
        }
        
        # Collect monitor data
        for name, monitor in self.monitors.items():
            df = monitor.get_metrics_dataframe()
            if not df.empty:
                report['monitors'][name] = {
                    'metrics_count': len(df),
                    'time_range': {
                        'start': df['timestamp'].min(),
                        'end': df['timestamp'].max()
                    },
                    'metric_types': df['type'].unique().tolist(),
                    'summary_stats': self._compute_summary_stats(df)
                }
                
        # Collect analyzer results
        for name, analyzer in self.analyzers.items():
            if hasattr(analyzer, 'analyze'):
                try:
                    results = analyzer.analyze()
                    report['analyzers'][name] = results
                except Exception as e:
                    report['analyzers'][name] = {'error': str(e)}
                    
        # Generate summary
        report['summary'] = self._generate_summary(report)
        
        # Format output
        if output_format == 'json':
            return json.dumps(report, indent=2, default=str)
        elif output_format == 'html':
            return self._generate_html_report(report)
        else:
            return report
    
    def _compute_summary_stats(self, df):
        """Compute summary statistics for metrics dataframe."""
        stats_dict = {}
        
        for metric_name in df['name'].unique():
            metric_data = df[df['name'] == metric_name]['value']
            stats_dict[metric_name] = {
                'mean': metric_data.mean(),
                'std': metric_data.std(),
                'min': metric_data.min(),
                'max': metric_data.max(),
                'count': len(metric_data)
            }
            
        return stats_dict
    
    def _generate_summary(self, report):
        """Generate executive summary."""
        summary = {}
        
        # Overall metrics
        total_metrics = sum(
            monitor_data.get('metrics_count', 0) 
            for monitor_data in report['monitors'].values()
        )
        summary['total_metrics_collected'] = total_metrics
        
        # Performance indicators
        if 'optimization' in report['monitors']:
            opt_data = report['monitors']['optimization']
            if 'summary_stats' in opt_data:
                stats = opt_data['summary_stats']
                summary['best_optimization_value'] = stats.get('best_value', {}).get('min')
                summary['average_evaluation_time'] = stats.get('evaluation_time', {}).get('mean')
                
        # Resource usage
        if 'system' in report['monitors']:
            sys_data = report['monitors']['system']
            if 'summary_stats' in sys_data:
                stats = sys_data['summary_stats']
                summary['peak_memory_usage'] = stats.get('process_memory_mb', {}).get('max')
                summary['average_cpu_usage'] = stats.get('cpu_percent', {}).get('mean')
                
        return summary
    
    def _generate_html_report(self, report):
        """Generate HTML report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; }
                .metric { margin: 10px 0; padding: 10px; border: 1px solid #ddd; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Performance Report</h1>
            <p>Generated: {timestamp}</p>
            
            <div class="section">
                <h2>Executive Summary</h2>
                {summary_html}
            </div>
            
            <div class="section">
                <h2>Monitor Details</h2>
                {monitors_html}
            </div>
            
            <div class="section">
                <h2>Analysis Results</h2>
                {analyzers_html}
            </div>
        </body>
        </html>
        """
        
        # Generate summary HTML
        summary = report.get('summary', {})
        summary_html = "<ul>"
        for key, value in summary.items():
            summary_html += f"<li><strong>{key}:</strong> {value}</li>"
        summary_html += "</ul>"
        
        # Generate monitors HTML
        monitors_html = ""
        for name, data in report.get('monitors', {}).items():
            monitors_html += f"<h3>{name}</h3>"
            monitors_html += f"<p>Metrics collected: {data.get('metrics_count', 0)}</p>"
            
            if 'summary_stats' in data:
                monitors_html += "<table>"
                monitors_html += "<tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>"
                for metric, stats in data['summary_stats'].items():
                    monitors_html += f"<tr><td>{metric}</td><td>{stats.get('mean', 'N/A'):.4f}</td>"
                    monitors_html += f"<td>{stats.get('std', 'N/A'):.4f}</td>"
                    monitors_html += f"<td>{stats.get('min', 'N/A'):.4f}</td>"
                    monitors_html += f"<td>{stats.get('max', 'N/A'):.4f}</td></tr>"
                monitors_html += "</table>"
                
        # Generate analyzers HTML
        analyzers_html = ""
        for name, data in report.get('analyzers', {}).items():
            analyzers_html += f"<h3>{name}</h3>"
            if 'error' in data:
                analyzers_html += f"<p>Error: {data['error']}</p>"
            else:
                analyzers_html += "<pre>" + json.dumps(data, indent=2) + "</pre>"
                
        return html.format(
            timestamp=report['timestamp'],
            summary_html=summary_html,
            monitors_html=monitors_html,
            analyzers_html=analyzers_html
        )
    
    def save_report(self, filename, output_format='dict'):
        """Save report to file."""
        report = self.generate_report(output_format)
        
        with open(filename, 'w') as f:
            if output_format == 'json':
                f.write(report)
            elif output_format == 'html':
                f.write(report)
            else:
                json.dump(report, f, indent=2, default=str)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        """Initialize performance benchmark."""
        self.benchmark_results = {}
        self.test_functions = {}
        
    def register_test_function(self, name, func, dimensions, difficulty='medium'):
        """
        Register a test function for benchmarking.
        
        Parameters
        ----------
        name : str
            Name of the test function
        func : callable
            Test function
        dimensions : int
            Number of dimensions
        difficulty : str, default='medium'
            Difficulty level: 'easy', 'medium', 'hard', 'ultra'
        """
        self.test_functions[name] = {
            'func': func,
            'dimensions': dimensions,
            'difficulty': difficulty
        }
        
    def benchmark_optimizer(self, optimizer_name, optimizer_class, 
                          optimizer_params=None, max_iterations=100):
        """
        Benchmark an optimizer on all test functions.
        
        Parameters
        ----------
        optimizer_name : str
            Name of the optimizer
        optimizer_class : class
            Optimizer class
        optimizer_params : dict, optional
            Parameters for optimizer
        max_iterations : int, default=100
            Maximum iterations per test
            
        Returns
        -------
        results : dict
            Benchmark results
        """
        if optimizer_name not in self.benchmark_results:
            self.benchmark_results[optimizer_name] = {}
            
        optimizer_params = optimizer_params or {}
        results = {}
        
        for test_name, test_info in self.test_functions.items():
            print(f"Benchmarking {optimizer_name} on {test_name}...")
            
            # Setup monitoring
            monitor = OptimizationPerformanceMonitor()
            monitor.start_optimization()
            
            # Run optimization
            start_time = time.time()
            
            try:
                # Create optimizer
                optimizer = optimizer_class(**optimizer_params)
                
                # Run optimization loop (simplified)
                best_value = float('inf')
                for i in range(max_iterations):
                    # Generate random point (simplified)
                    x = np.random.uniform(-5, 5, test_info['dimensions'])
                    
                    # Evaluate
                    eval_start = time.time()
                    y = test_info['func'](x)
                    eval_time = time.time() - eval_start
                    
                    # Record metrics
                    monitor.record_iteration(x, y, eval_time)
                    
                    if y < best_value:
                        best_value = y
                        
                total_time = time.time() - start_time
                
                # Store results
                results[test_name] = {
                    'best_value': best_value,
                    'total_time': total_time,
                    'iterations': max_iterations,
                    'success': True,
                    'metrics': monitor.get_metrics_dataframe().to_dict('records')
                }
                
            except Exception as e:
                results[test_name] = {
                    'error': str(e),
                    'success': False
                }
                
        self.benchmark_results[optimizer_name] = results
        return results
    
    def compare_optimizers(self, metric='best_value'):
        """
        Compare optimizers across all test functions.
        
        Parameters
        ----------
        metric : str, default='best_value'
            Metric to compare
            
        Returns
        -------
        comparison : pd.DataFrame
            Comparison results
        """
        comparison_data = []
        
        for optimizer_name, results in self.benchmark_results.items():
            for test_name, result in results.items():
                if result.get('success', False):
                    comparison_data.append({
                        'optimizer': optimizer_name,
                        'test_function': test_name,
                        'metric_value': result.get(metric, np.nan),
                        'total_time': result.get('total_time', np.nan)
                    })
                    
        return pd.DataFrame(comparison_data)
    
    def generate_benchmark_report(self):
        """Generate comprehensive benchmark report."""
        if not self.benchmark_results:
            return "No benchmark results available."
            
        report = "# Performance Benchmark Report\n\n"
        report += f"Generated: {datetime.now()}\n\n"
        
        # Summary table
        comparison = self.compare_optimizers()
        if not comparison.empty:
            report += "## Performance Comparison\n\n"
            report += comparison.to_string(index=False)
            report += "\n\n"
            
        # Detailed results
        report += "## Detailed Results\n\n"
        for optimizer, results in self.benchmark_results.items():
            report += f"### {optimizer}\n\n"
            
            for test_name, result in results.items():
                report += f"**{test_name}:**\n"
                if result.get('success', False):
                    report += f"- Best value: {result.get('best_value', 'N/A'):.6f}\n"
                    report += f"- Total time: {result.get('total_time', 'N/A'):.4f}s\n"
                    report += f"- Iterations: {result.get('iterations', 'N/A')}\n"
                else:
                    report += f"- Error: {result.get('error', 'Unknown')}\n"
                report += "\n"
                
        return report


# Utility functions
def create_performance_suite():
    """Create a complete performance monitoring suite."""
    suite = PerformanceReporter()
    
    # Add monitors
    suite.add_monitor('system', SystemResourceMonitor())
    suite.add_monitor('optimization', OptimizationPerformanceMonitor())
    
    # Add analyzers
    suite.add_monitor('convergence', ConvergenceAnalyzer())
    suite.add_monitor('scalability', ScalabilityAnalyzer())
    
    return suite


def profile_function(func, *args, **kwargs):
    """
    Profile a function's performance.
    
    Parameters
    ----------
    func : callable
        Function to profile
    *args, **kwargs
        Arguments for the function
        
    Returns
    -------
    result : tuple
        (function_result, profiling_data)
    """
    # Start monitoring
    monitor = SystemResourceMonitor()
    monitor.start_monitoring()
    
    # Execute function
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    try:
        result = func(*args, **kwargs)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
        
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Collect profiling data
    profiling_data = {
        'execution_time': end_time - start_time,
        'memory_delta': (end_memory - start_memory) / (1024**2),  # MB
        'success': success,
        'error': error,
        'system_metrics': monitor.get_metrics_dataframe().to_dict('records')
    }
    
    return result, profiling_data


# Test functions for benchmarking
def sphere_function(x):
    """Sphere test function."""
    return np.sum(x**2)

def rastrigin_function(x):
    """Rastrigin test function."""
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock_function(x):
    """Rosenbrock test function."""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


if __name__ == "__main__":
    # Example usage
    print("Ultra-Advanced Performance Analytics Module")
    print("=" * 50)
    
    # Create performance suite
    suite = create_performance_suite()
    
    # Start monitoring
    suite.monitors['system'].start_monitoring()
    
    # Simulate optimization
    opt_monitor = suite.monitors['optimization']
    opt_monitor.start_optimization()
    
    for i in range(20):
        x = np.random.randn(5)
        y = sphere_function(x)
        opt_monitor.record_iteration(x, y)
        time.sleep(0.1)
        
    # Generate report
    report = suite.generate_report()
    print("\nPerformance Report Summary:")
    print(json.dumps(report['summary'], indent=2))
    
    # Benchmark example
    print("\nRunning benchmark...")
    benchmark = PerformanceBenchmark()
    benchmark.register_test_function('sphere', sphere_function, 5, 'easy')
    benchmark.register_test_function('rastrigin', rastrigin_function, 5, 'hard')
    
    # Note: In practice, you would benchmark actual optimizers
    print("Benchmark setup complete. Ready to test optimizers.")
    
    # Stop monitoring
    suite.monitors['system'].stop_monitoring()
