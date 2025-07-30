import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


class Quality:
    def __init__(
        self,
        dir_path: str,
        dataset: str = None,
        main_color: str = 'blue',
        second_color: str = 'cyan',
        linewidth: float = 0.75,
        grid_alpha: float = 0.25,
        save_path: str = ''
    ):
        """
        Class for loading and analyzing quality datasets.
        """
        self.dataname = ''
        self.save_path = save_path
        self.linewidth = linewidth
        self.grid_alpha = grid_alpha
        self.main_color = main_color
        self.second_color = second_color

        # Load dataset from single file or multiple files in a directory
        if dataset:
            self.dataname = f"_{dataset[:-5]}"
            with open(os.path.join(dir_path, dataset), 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = self._load_multiple_json(dir_path)

        # Assign loaded variables as attributes
        for key, value in data.items():
            setattr(self, key, value)

        # Generate timestamp attribute
        self._build_timestamp()

    def _load_multiple_json(self, dir_path: str) -> dict:
        """
        Load all JSON files from a directory and merge their data.
        """
        data = {}
        self.sampling_n = {}

        for file in os.listdir(dir_path):
            if not file.endswith('.json'):
                continue

            with open(os.path.join(dir_path, file), 'r', encoding='utf-8') as f:
                content = json.load(f)

            self.sampling_n[file[:-5]] = len(content['MW']['data'])

            for key, val in content.items():
                if key not in data:
                    data[key] = {
                        'data': val['data'],
                        'Rate': val.get('Rate'),
                        'Description': val.get('Description', ''),
                        'Units': val.get('Units', '')
                    }
                else:
                    data[key]['data'].extend(val['data'])

        return data

    def _build_timestamp(self):
        """
        Create datetime64 timestamp series based on date/time variables.
        """
        expander = self.GMT_SEC['Rate'] // self.DATE_YEAR['Rate']

        y = np.repeat(self.DATE_YEAR['data'], expander)
        m = np.repeat(self.DATE_MONTH['data'], expander)
        d = np.repeat(self.DATE_DAY['data'], expander)
        h = np.array(self.GMT_HOUR['data'])
        mi = np.array(self.GMT_MINUTE['data'])
        s = np.array(self.GMT_SEC['data'])

        timestamps = []
        for yi, mi_, di, hi, mii, si in zip(y, m, d, h, mi, s):
            try:
                if (1 <= mi_ <= 12 and 1 <= di <= 31 and
                    0 <= hi <= 23 and 0 <= mii <= 59 and 0 <= si < 60 and
                    1900 <= yi <= datetime.now().year + 1):

                    dt = datetime(int(yi), int(mi_), int(di), int(hi), int(mii), int(si))
                    timestamps.append(np.datetime64(dt))
                else:
                    timestamps.append(np.datetime64('NaT'))
            except Exception:
                timestamps.append(np.datetime64('NaT'))

        self.TIMESTAMP = {
            'data': timestamps,
            'Rate': self.GMT_SEC['Rate'],
            'Units': 'DATETIME'
        }

    def dataframe(self, variables: list) -> pd.DataFrame:
        """
        Build a pandas DataFrame with matched sampling rates.
        """
        rates = {var: getattr(self, var)['Rate'] for var in variables}
        max_rate = max(rates.values())

        df_dict = {}
        for var in variables:
            values = getattr(self, var)['data']
            repeat_factor = max_rate // rates[var]
            df_dict[var] = np.repeat(values, repeat_factor)

        return pd.DataFrame(df_dict)

    def time_series(self, var: str, time_limits: list = None):
        """
        Plot a time series for a selected variable.
        """
        df = self.dataframe([var, 'TIMESTAMP'])
        x = df['TIMESTAMP']
        y = df[var]

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(x, y, linewidth=self.linewidth, color=self.main_color)
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{var}, {getattr(self, var)['Units']}")
        ax.grid(alpha=self.grid_alpha)

        valid = ~np.isnat(x)
        min_time = pd.to_datetime(np.min(x[valid])).strftime("%Y-%m-%d %H:%M:%S")
        max_time = pd.to_datetime(np.max(x[valid])).strftime("%Y-%m-%d %H:%M:%S")

        # Title and x-limits depending on user input
        if time_limits:
            start = pd.Timestamp(time_limits[0])
            end = pd.Timestamp(time_limits[1]) if len(time_limits) > 1 else np.max(x[valid])
            ax.set_xlim([start, end])
            ax.set_title(f"{getattr(self, var)['Description']} Time Series: {start} - {end}")
        else:
            ax.set_title(f"{getattr(self, var)['Description']} Time Series: {min_time} - {max_time}")

        # Save figure
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"ts_{var}_{now}{self.dataname}.png"
        plt.savefig(os.path.join(self.save_path, filename))
        plt.close()

    def correlation(self, X: str, Y: str, group: str = 'PH', save_df: bool = False):
        """
        Compute and plot Pearson correlation per group.
        """
        vars_list = [X, Y] + ([group] if group else [])
        df = self.dataframe(vars_list)

        groups = df[group].unique() if group else [None]
        fig, axes = plt.subplots(len(groups), 1, figsize=(16, 9 * len(groups)))
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        for ax, g in zip(axes, groups):
            subset = df[df[group] == g] if group else df
            r, _ = pearsonr(subset[X], subset[Y])
            ax.scatter(subset[X], subset[Y], color=self.main_color)
            ax.set_title(f"{getattr(self, Y)['Description']} x {getattr(self, X)['Description']} for {group} {g} - r = {r:.3f}")
            ax.set_xlabel(f"{X}, {getattr(self, X)['Units']}")
            ax.set_ylabel(f"{Y}, {getattr(self, Y)['Units']}")
            ax.grid(alpha=self.grid_alpha)

        plt.tight_layout()
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"corr_{Y}_{X}_{now}{self.dataname}"
        plt.savefig(os.path.join(self.save_path, filename + '.png'))
        plt.close()

        if save_df:
            df.to_excel(os.path.join(self.save_path, filename + '.xlsx'), index=False)

    def shewhart_chart(self, var: str, group: str = 'PH', series: str = 'DATE_DAY', save_df: bool = False):
        """
        Generate Shewhart control charts by group and time series.
        """
        df = self.dataframe([var, group, series])
        grouped = df.groupby([group, series]).agg(
            mean=(var, 'mean'),
            std=(var, 'std'),
            count=(var, 'count')
        )
        self.grouped_df = grouped

        group_keys = sorted(set(idx[0] for idx in grouped.index))
        fig, axes = plt.subplots(len(group_keys), 1, figsize=(16, 9 * len(group_keys)))
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        for ax, g in zip(axes, group_keys):
            subset = grouped.xs(g, level=group)
            xbarbar = subset['mean'].mean()
            sigmabar = subset['mean'].std()
            n_series = len(subset)
            CL = sigmabar / np.sqrt(n_series)

            ax.plot(subset.index.values, subset['mean'], marker='o', label='X̄', color='k', linewidth=self.linewidth)
            ax.axhline(y=xbarbar, linestyle='--', linewidth=self.linewidth, color='k', label='Overall Mean')

            xmin, xmax = ax.get_xlim()
            x_fill = np.linspace(xmin, xmax, 500)

            ax.fill_between(x_fill, xbarbar - CL, xbarbar + CL, color='green', alpha=self.grid_alpha, label='Zone C')
            ax.fill_between(x_fill, xbarbar + CL, xbarbar + 2 * CL, color='orange', alpha=self.grid_alpha, label='Zone B')
            ax.fill_between(x_fill, xbarbar - 2 * CL, xbarbar - CL, color='orange', alpha=self.grid_alpha)
            ax.fill_between(x_fill, xbarbar + 2 * CL, xbarbar + 3 * CL, color='red', alpha=self.grid_alpha, label='Zone A')
            ax.fill_between(x_fill, xbarbar - 3 * CL, xbarbar - 2 * CL, color='red', alpha=self.grid_alpha)

            ax.set_xlim([xmin, xmax])
            ax.set_xlabel(f"{series}, {getattr(self, series)['Units']}")
            ax.set_ylabel(f"{var} Mean, {getattr(self, var)['Units']}")
            ax.set_title(f"{getattr(self, var)['Description']} vs {getattr(self, series)['Description']}\nShewhart Chart for {group} {g}")
            ax.grid(alpha=self.grid_alpha)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), edgecolor='k')

        plt.tight_layout()
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"sc_{var}_{series}_{now}{self.dataname}"
        plt.savefig(os.path.join(self.save_path, filename + '.png'))
        plt.close()

        if save_df:
            df.to_excel(os.path.join(self.save_path, filename + '.xlsx'), index=False)