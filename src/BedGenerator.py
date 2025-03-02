import numpy as np
import pandas as pd
import re
import os
from scipy.stats import kurtosis, skew, johnsonsu, genextreme, norm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


class VCFFileNotFoundError(Exception):
    """
    Custom exception which raises when
    VCF file not found.
    """

    def __init__(self, vcf_file):
        super().__init__(f"VCF file '{vcf_file}' not found.")


class ChromosomeNotFoundError(Exception):
    """
    Custom exception which raises when length for
    chromosome not found.
    """

    def __init__(self, chromosome):
        super().__init__(
            f"Chromosome '{chromosome}' not found in chromosomal lengths dictionary."
        )


class BreakpointExtractor:
    """
    The class for extracting breakpoint positions
    from a VCF file for the specified chromosomes.

    This class realized methods to read a VCF file
    and extracts breakpoint positions for
    structural variations such as deletions (SVTYPE=DEL)
    and breakends (SVTYPE=BND), but only for homozygous genotypes with
    alternate allele (genotype starts with "1/1"). The breakpoint positions
    are stored for each of the specified chromosomes.

    Attributes:

    :param vcf_file: str: path to the VCF-file.
    :param chromosomal_lengths: dict: dictionary mapping
    chromosome names to their lengths.
    :param breakpoints: dict: dictionary storing breakpoint
    positions for each chromosome.
    """

    def __init__(self, vcf_file: str, chromosomal_lengths: dict[str, int] = None):
        """
        Initializes the BreakpointPositionsExtractor
        with a given VCF file and optional chromosome lengths.

        Args:
        - vcf_file: str: path to the VCF-file.
        - chromosomal_lengths: dict: a dictionary containing
        chromosome names as keys and their respective lengths
        as values. If not provided, default human chromosome
        lengths are used.
        """
        self.vcf_file = vcf_file
        self.chromosomal_lengths = (
            chromosomal_lengths
            if chromosomal_lengths
            else {
                "1": 248956422,
                "2": 242193529,
                "3": 198295559,
                "4": 190214555,
                "5": 181538259,
                "6": 170805979,
                "7": 159345973,
                "8": 145138636,
                "9": 138394717,
                "10": 133797422,
                "11": 135086622,
                "12": 133275309,
                "13": 114364328,
                "14": 107043718,
                "15": 101991189,
                "16": 90338345,
                "17": 83257441,
                "18": 80373285,
                "19": 58617616,
                "20": 64444167,
                "21": 46709983,
                "22": 50818468,
                "X": 156040895,
                "Y": 57227415,
            }
        )
        self.breakpoints = {chrom: [] for chrom in self.chromosomal_lengths}
        self.fragment_lengths = {}
        self.fragment_lengths_normalized = {}

    def _parse_vcf_file(self) -> None:
        """
        Parses the VCF file and extracts breakpoint
        positions for each chromosome.

        This method processes each line of the VCF
        file, filtering out comments, and extracts
        positions of structural variations
        (deletions and breakends) for homozygous
        variants (genotype '1/1').

        :raises: Exception: when VCF file not found.
        """
        if not os.path.exists(self.vcf_file):
            raise VCFFileNotFoundError(self.vcf_file)

        try:
            with open(self.vcf_file, "r") as vcf:
                for line in vcf:
                    if line.startswith("#"):
                        continue
                    fields = line.strip().split("\t")
                    chrom = fields[0]
                    pos = int(fields[1])
                    info_field = fields[7]
                    genotype_info = fields[9]

                    valid_chromosomes = {str(i) for i in range(1, 23)} | {"X", "Y"}
                    if chrom not in valid_chromosomes:
                        continue  # Пропускаем ненужные хромосомы

                    if genotype_info.startswith("1/1") and re.search(
                        r"SVTYPE=(DEL|BND)", info_field
                    ):
                        self.breakpoints[chrom].append(pos)

        except FileNotFoundError:
            raise VCFFileNotFoundError(self.vcf_file)

    def get_breakpoints_positions(self) -> dict[str, list[int]]:
        """
        Extracts and returns a dictionary of sorted
        unique breakpoint positions for each chromosome.

        :return: dict: a dictionary where keys are chromosome
        names and values are sorted lists of unique breakpoint
        positions.
        """
        self._parse_vcf_file()
        for chrom in self.breakpoints:
            self.breakpoints[chrom] = sorted(set(self.breakpoints[chrom]))
        return self.breakpoints

    def get_fragments_length(
        self,
    ) -> tuple[dict[str, list[int]], dict[str, list[float]]]:
        """
        Calculates fragment lengths based on the provided
        breakpoint positions and chromosome lengths.

        For each chromosome, this function calculates the
        fragment lengths, which are the distances between
        consecutive breakpoints. It also computes the normalized
        fragment lengths, where the normalization
        is performed using the chromosome length. The normalization
        is done by taking the negative logarithm
        of the fragment length as a fraction of the chromosome length.
        """
        self._parse_vcf_file()
        for chrom, breakpoints in self.breakpoints.items():
            if chrom not in self.chromosomal_lengths:
                raise ChromosomeNotFoundError(chrom)

            chrom_length = self.chromosomal_lengths[chrom]
            fragments = []

            if breakpoints:
                fragments.append(breakpoints[0])
                for i in range(1, len(breakpoints)):
                    fragments.append(breakpoints[i] - breakpoints[i - 1])
                fragments.append(chrom_length - breakpoints[-1])
            else:
                fragments.append(0)

            self.fragment_lengths[chrom] = fragments

            normalized_fragments = [
                -np.log10(length / chrom_length) if length > 0 else np.nan
                for length in fragments
            ]

            self.fragment_lengths_normalized[chrom] = normalized_fragments

        return self.fragment_lengths, self.fragment_lengths_normalized

    def get_bins_of_fragments_normalized(self) -> dict:
        """
        Bins normalized fragment lengths into logarithmically
        spaced bins for each chromosome.

        This function calculates the number of fragments falling
        into each logarithmic bin for each chromosome
        based on the provided normalized fragment lengths.
        The fragment lengths are binned into logarithmic intervals
        to handle a wide range of values. The function creates
        logarithmic bins based on the distribution of fragment
        lengths across all chromosomes, counts the occurrences of
        fragment lengths in each bin, and returns the results
        in a dictionary.

        Parameters:
        ----------
        fragment_lengths_normalized : dict
            A dictionary where keys are chromosome identifiers
            (e.g., '1', '2', 'X') and values are lists of normalized
            fragment lengths for each chromosome. These normalized
            lengths are calculated using the negative log of the
            fragment length relative to the chromosome length.

        Returns:
        -------
        fragments_bins : dict
            A dictionary where keys are chromosome identifiers
            and values are dictionaries containing the count of
            fragment lengths that fall into each logarithmic bin.
            The inner dictionary has bin labels as keys (e.g.,
            '0.10-0.20') and the count of fragments in that bin as values.

        Notes:
        ------
        - The number of bins is determined using Sturges' formula:
        1 + log2(n), where n is the total number of fragment lengths
        across all chromosomes.
        - The bin edges are logarithmically spaced, which allows for
        handling fragment lengths that span a wide range of values.
        - The function filters out non-finite fragment lengths
        (e.g., `np.inf` or NaN values) during the binning process.
        """
        self.fragments_bins_normalized = {}

        total_fragment_lengths = []
        for fragment_lengths in self.fragment_lengths_normalized.values():
            total_fragment_lengths.extend(fragment_lengths)

        total_fragment_lengths = [
            length for length in total_fragment_lengths if np.isfinite(length)
        ]

        n = len(total_fragment_lengths)
        if n == 0:
            print("No valid fragment lengths available for binning.")
            return {}

        bin_count = int(np.ceil(1 + np.log2(n)))
        bins = np.logspace(
            np.log10(min(total_fragment_lengths)),
            np.log10(max(total_fragment_lengths)),
            num=bin_count,
        )

        bin_labels = [
            f"{float(bins[i]):.2f}-{float(bins[i+1]):.2f}" for i in range(len(bins) - 1)
        ]

        for chrom, lengths in self.fragment_lengths_normalized.items():
            bin_count_dict = {label: 0 for label in bin_labels}
            for length in lengths:
                for i in range(len(bins) - 1):
                    if bins[i] <= length < bins[i + 1]:
                        bin_label = f"{float(bins[i]):.2f}-{float(bins[i+1]):.2f}"
                        bin_count_dict[bin_label] += 1
                        break

            self.fragments_bins_normalized[chrom] = bin_count_dict

        return self.fragments_bins_normalized

    def get_bins_of_fragments_not_normalized(self) -> dict:
        """ """
        self.fragment_bins_not_normalized = {}

        for chrom, lengths in self.fragment_lengths.items():
            if not lengths:
                continue

            max_length = max(lengths)
            bin_edges = list(range(1, min(max_length, 10000000) + 100000, 100000)) + [
                float("inf")
            ]

            bin_labels = [
                f"{bin_edges[i]}-{bin_edges[i+1] - 1}"
                if bin_edges[i + 1] != float("inf")
                else "Max lenght"
                for i in range(len(bin_edges) - 1)
            ]

            bin_count_dict = {label: 0 for label in bin_labels}

            for length in lengths:
                for i in range(len(bin_edges) - 1):
                    if bin_edges[i] <= length < bin_edges[i + 1]:
                        bin_label = (
                            f"{bin_edges[i]}-{bin_edges[i+1] - 1}"
                            if bin_edges[i + 1] != float("inf")
                            else "Max lenght"
                        )
                        bin_count_dict[bin_label] += 1
                        break

            total = 0
            found = False
            for label in bin_labels:
                if bin_count_dict[label] <= 0 and not found:
                    total += bin_count_dict[label]
                    bin_count_dict[label] = 0
                    found = True
                elif found:
                    total += bin_count_dict[label]
                    bin_count_dict[label] = 0

            if found:
                bin_count_dict["Max lenght"] = total

            keys_to_remove = [
                label
                for label in bin_labels
                if bin_count_dict[label] == 0 and label != "Max lenght"
            ]
            for key in keys_to_remove:
                del bin_count_dict[key]

            self.fragment_bins_not_normalized[chrom] = bin_count_dict

        return self.fragment_bins_not_normalized

    def plot_fragment_length_histogram_by_chromosome_normalized(
        self, output_directory: str
    ) -> None:
        """
        Plots the distribution of normalized fragment lengths
        in logarithmic bins for each chromosome.

        This function generates bar plots to visualize the
        distribution of normalized fragment lengths
        across different bins for each chromosome. It handles
        each chromosome separately, filtering out
        bins with zero counts and adjusting the x-axis labels
        for readability. The plots are saved as PNG files
        in the specified output directory.

        Parameters:
        ----------
        fragments_bins : dict
            A dictionary where keys are chromosome identifiers
            (e.g., '1', '2', 'X') and values are dictionaries
            containing the count of fragments that fall into each
            logarithmic bin. The inner dictionary has bin labels
            as keys (e.g., '0.10-0.20') and the count of fragments
            in that bin as values.

        output_directory : str
            The path to the directory where the generated plots
            will be saved as PNG files. The plots will be saved
            using filenames based on the chromosome
            (e.g., 'distribution_chromosome_1.png').

        Returns:
        -------
        None
            This function does not return any value. It saves the
            plots as PNG files to the specified directory.

        Notes:
        ------
        - The function processes each chromosome's fragment length distribution independently.
        - The x-axis labels are adjusted for readability and formatted as ranges (e.g., '0.10-0.20').
        - The plots are saved in the provided directory with filenames indicating the chromosome.
        """
        output_dir = output_directory + "plots_histograms_normalized"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        rows = []
        for chrom, bins_count in self.fragments_bins_normalized.items():
            for bin_range, count in bins_count.items():
                rows.append(
                    {"Chromosome": chrom, "Bin Range": bin_range, "Count": count}
                )

        data_bins_distribution_by_chromosome = pd.DataFrame(rows)

        unique_chromosomes = data_bins_distribution_by_chromosome["Chromosome"].unique()

        for chrom in unique_chromosomes:
            data_chrom = data_bins_distribution_by_chromosome[
                data_bins_distribution_by_chromosome["Chromosome"] == chrom
            ]
            first_nonzero_idx = data_chrom[data_chrom["Count"] > 0].index.min()

            if not pd.isna(first_nonzero_idx):
                data_chrom_nonzero = data_chrom.loc[first_nonzero_idx:]
            else:
                data_chrom_nonzero = data_chrom

            plt.figure(figsize=(10, 6))
            sns.barplot(
                x="Bin Range", y="Count", data=data_chrom_nonzero, palette="Purples_d"
            )
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            bin_labels = data_chrom_nonzero["Bin Range"].unique()
            new_bin_labels = [
                f'{float(label.split("-")[0])}-{float(label.split("-")[1])}'
                for label in bin_labels
            ]
            plt.xticks(
                ticks=range(len(bin_labels)),
                labels=new_bin_labels,
                rotation=45,
                ha="right",
                fontsize=10,
            )
            plt.yticks(fontsize=12)
            plt.title(
                f"Распределение нормализованных длин фрагментов для хромосомы {chrom}",
                fontsize=14,
            )
            plt.xlabel(
                "Нормализованные диапазоны бинов для фрагментов хромосомы", fontsize=12
            )
            plt.ylabel("Количество фрагментов", fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/distribution_chromosome_{chrom}.png")
            plt.close()

    def plot_fragment_length_histogram_by_chromosome_not_normalized(
        self, output_directory: str
    ) -> None:
        """ """
        output_dir = output_directory + "plots_histograms_not_normalized"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        rows = []
        for chrom, bins_count in self.fragment_bins_not_normalized.items():
            for bin_range, count in bins_count.items():
                rows.append(
                    {"Chromosome": chrom, "Bin Range": bin_range, "Count": count}
                )

        data_bins_distribution_by_chromosome = pd.DataFrame(rows)
        unique_chromosomes = data_bins_distribution_by_chromosome["Chromosome"].unique()

        for chrom in unique_chromosomes:
            data_chrom = data_bins_distribution_by_chromosome[
                data_bins_distribution_by_chromosome["Chromosome"] == chrom
            ]
            first_nonzero_idx = data_chrom[data_chrom["Count"] > 0].index.min()

            if not pd.isna(first_nonzero_idx):
                data_chrom_nonzero = data_chrom.loc[first_nonzero_idx:]
            else:
                data_chrom_nonzero = data_chrom

            plt.figure(figsize=(10, 6))
            sns.barplot(
                x="Bin Range", y="Count", data=data_chrom_nonzero, palette="Purples_d"
            )
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            bin_labels = data_chrom_nonzero["Bin Range"].unique()
            new_bin_labels = []

            for label in bin_labels:
                if label != "Max lenght":
                    new_bin_labels.append(
                        f'{float(label.split("-")[0])}-{float(label.split("-")[1])}'
                    )
                else:
                    new_bin_labels.append(label)
            plt.xticks(
                ticks=range(len(bin_labels)),
                labels=new_bin_labels,
                rotation=45,
                ha="right",
                fontsize=10,
            )
            plt.yticks(fontsize=12)
            plt.title(
                f"Распределение не нормализованных длин фрагментов для хромосомы {chrom}",
                fontsize=14,
            )
            plt.xlabel(
                "Не нормализованные диапазоны бинов для фрагментов хромосомы",
                fontsize=12,
            )
            plt.ylabel("Количество фрагментов", fontsize=12)
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    output_dir, f"{chrom}_fragment_length_histogram_not_normalized.png"
                )
            )
            plt.close()

    def __repr__(self) -> str:
        """
        Returns a formatted string representation of
        breakpoints and fragment lengths.
        """
        breakpoints_repr = "\n".join(
            [
                f"Chromosome {k}: Length: {len(v)}"
                for k, v in self.breakpoints.items()
                if v
            ]
        )
        fragment_lengths_repr = "\n".join(
            [
                f"Chromosome {k}: Length: {len(v)}"
                for k, v in self.fragment_lengths.items()
                if v
            ]
        )
        fragment_lengths_normalized_repr = "\n".join(
            [
                f"Chromosome {k}: Length: {len(v)}"
                for k, v in self.fragment_lengths_normalized.items()
                if v
            ]
        )

        return (
            f"Breakpoints:\n{breakpoints_repr}\n\n"
            f"Fragment lengths:\n{fragment_lengths_repr}\n\n"
            f"Fragment lengths normalized:\n{fragment_lengths_normalized_repr}"
        )


class FragmentStatistics(BreakpointExtractor):
    """
    This class for computing statistical properties
    of fragment lengths extracted from breakpoints.
    """

    def __init__(self, vcf_file, chromosomal_lengths=None):
        """
        Inheritance from parent class BreakpointExtractor.
        """
        super().__init__(vcf_file, chromosomal_lengths)
        self.stats = None

    def get_fragment_stats(self, save_dir: str = "./"):
        """
        Computes statistical measures for fragment lengths:
        - percentiles (25, 75)
        - median
        - sd
        - variance
        - kurtosis
        - skewness

        :param save_dir: str: Directory where the statistics CSV file will be saved
        :return: pd.DataFrame: DataFrame containing computed statistics for each chromosome.
        """
        self.get_fragments_length()

        self.stats = []
        for chrom, lengths in self.fragment_lengths_normalized.items():
            valid_lengths = np.array(lengths)
            valid_lengths = valid_lengths[~np.isnan(valid_lengths)]
            if len(valid_lengths) == 0:
                continue

            q1, median, q3 = np.percentile(valid_lengths, [25, 50, 75])
            sd = np.std(valid_lengths)
            variance = np.var(valid_lengths)
            kurt = kurtosis(valid_lengths)
            skewness = skew(valid_lengths)

            self.stats.append([chrom, median, q1, q3, kurt, skewness, sd, variance])

        self.df_stats = pd.DataFrame(
            self.stats,
            columns=[
                "Chromosome",
                "Median",
                "Q1",
                "Q3",
                "Kurtosis",
                "Skewness",
                "SD",
                "Variance",
            ],
        )

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "fragment_stats.csv")
        self.df_stats.to_csv(save_path, index=False)

        return self.df_stats


class BedGenerator(FragmentStatistics):
    """
    Class for generating BED-like file output
    from normalized breakpoint fragments.

    This class processes structural variation
    breakpoints and generates fragment lengths
    fitted to statistical distributions, which are then
    used to create a BED file-like output.
    """

    def __init__(self, vcf_file: str, chromosomal_lengths: dict[str, int] = None):
        """
        Initializes the BedGenerator.

        Args:
        :param vcf_file: str: path to the VCF-file.
        :param chromosomal_lengths: dict[str, int]: dictionary
        with chromosome names as keys and their respective
        lengths as values. Defaults to None.
        """
        super().__init__(vcf_file, chromosomal_lengths)

    def _fit_distribution(self, lengths: list[float]) -> tuple:
        """
        Fits the best statistical distribution to
        the given normalized fragment lengths.

        Args:
        :param lengths: list[float]: list of normalized fragment lengths.

        Returns:
            - The best-fitting distribution (`scipy.stats` distribution object).
            - The parameters of the distribution.
        Returns (None, None) if there are no valid fragment lengths.
        """
        lengths = np.array(lengths)
        lengths = lengths[~np.isnan(lengths)]
        if len(lengths) == 0:
            return None, None

        mean, std = np.mean(lengths), np.std(lengths)
        skewness, kurt = skew(lengths), kurtosis(lengths)

        if abs(skewness) < 0.1 and abs(kurt) < 3.5:
            return norm, (mean, std)
        elif abs(skewness) > 0.1 and abs(kurt) > 3.5:
            shape1, shape2, loc, scale = johnsonsu.fit(lengths)
            return johnsonsu, (shape1, shape2, loc, scale)
        else:
            shape, loc, scale = genextreme.fit(lengths)
            return genextreme, (shape, loc, scale)

    def generate_normalized_fragments(self) -> dict[str, np.array]:
        """
        Generates normalized fragment lengths based
        on fitted distributions.

        :return dict[str, np.ndarray]: a dictionary where keys
        are chromosome names, and values are arrays of generated
        fragment lengths.
        """
        self.generated_fragments = {}
        for chrom, norm_lengths in self.fragment_lengths_normalized.items():
            if not norm_lengths or all(np.isnan(norm_lengths)):
                continue

            dist, params = self._fit_distribution(norm_lengths)
            if dist is None:
                continue

            size = len(norm_lengths)
            self.generated_fragments[chrom] = dist.rvs(*params, size=size)

        return self.generated_fragments

    def _restore_absolute_lengths(self) -> dict[str, np.array]:
        """
        Converts generated normalized fragment lengths
        back to absolute fragment lengths.

        :return dict[str, np.ndarray]: a dictionary where keys
        are chromosome names, and values are arrays of
        absolute fragment lengths.
        """
        self.absolute_lengths = {}
        self.generate_normalized_fragments()

        for chrom, norm_lengths in self.generated_fragments.items():
            chrom_length = self.chromosomal_lengths.get(chrom, 0)
            if chrom_length == 0:
                continue

            abs_lengths = (10 ** -np.array(norm_lengths)) * chrom_length
            abs_lengths = np.round(abs_lengths).astype(int)
            abs_lengths = abs_lengths[abs_lengths > 0]
            if len(abs_lengths) == 0:
                continue

            self.absolute_lengths[chrom] = abs_lengths

        return self.absolute_lengths

    def generate_bed(self, output_bed: str = None) -> pd.DataFrame:
        """
        Generates a BED-like file based on restored fragment lengths.

        :return pd.DataFrame: a DataFrame representing the
        BED file format with columns:
            - "chr": Chromosome name.
            - "start": Start position of the fragment.
            - "end": End position of the fragment.
        """
        self.get_fragments_length()
        print(f"Generating BED file: {output_bed} ...")
        self._restore_absolute_lengths()
        bed_records = []

        for chrom, lengths in self.absolute_lengths.items():
            chrom_length = self.chromosomal_lengths.get(chrom, 0)
            if chrom_length == 0:
                continue

            start = 0
            for length in lengths:
                end = start + length
                if end > chrom_length:
                    break

                bed_records.append([chrom, start, end])
                start = end

        bed_df = pd.DataFrame(bed_records, columns=["chr", "start", "end"])

        # Сохранение в файл в формате BED
        bed_df.to_csv(output_bed, sep="\t", index=False, header=False)
        print(f"BED file saved: {output_bed}")

        return bed_df
