import os
import numpy as np
import pandas as pd

rnastructure_path = "/Users/yvesmartin/lib/RNAstructure/exe"

def run_command(cmd):
    import subprocess

    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output.decode("utf-8")


class RNAstructure(object):

    """RNAstructure wrapper."""

    def __init__(self) -> None:
        pass

    def predict_partition(self, temperature_k=None, dms=None):
        # predict the partition of rna structures
        cmd = f"{os.path.join(rnastructure_path, 'partition')} {self.fasta_file} {self.pfs_file}"
        if temperature_k != None:
            cmd += " --temperature " + str(temperature_k)
        if type(dms) != type(None):
            assert len(self.sequence) == len(
                dms
            ), "The length of the sequence is not the same as the length of the signal."
            assert type(dms) in [
                list,
                tuple,
                np.ndarray,
            ], "The dms signal should be a list of floats."
            self.__write_dms_to_file(self.sequence, dms)
            cmd += " --shape " + self.dms_file
        run_command(cmd)

        # sum it into pairing probability
        run_command(
            os.path.join(rnastructure_path, "ProbabilityPlot")
            + " "
            + self.pfs_file
            + " -t "
            + self.prob_file
        )

        # read the probability file
        with open(self.prob_file, "r") as f:
            lines = f.readlines()
            pairingPrediction = {"i": [], "j": [], "p": []}
            for x in range(len(lines)):
                if x > 1:
                    ls = lines[x].split("\t")
                    pairingPrediction["i"] += [int(ls[0])]
                    pairingPrediction["j"] += [int(ls[1])]
                    pairingPrediction["p"] += [float(ls[2])]
        df = pd.DataFrame(pairingPrediction)

        # turn log probability into probability
        df["p"] = df["p"].apply(lambda x: np.power(10, -x))

        # add the reverse complement so that when we sum, we get the probability of each base being paired with anyone
        df = pd.concat([df, df.rename(columns={"i": "j", "j": "i"})])

        # cast: two bases being paired together => probability for each base of being paired with anyone
        df = df.groupby(["i"]).sum().drop(columns=["j"])

        # add p=0 for bases that are not paired with anyone
        df = df.reindex(range(1, len(self.sequence) + 1), fill_value=0)

        # Logs sum can lead to a probability > 1, but it should never be > 1.05. So we'll sanity check that, then cap it at 1.
        assert (df["p"] >= 0).all() and (
            df["p"] <= 1.05
        ).all(), "The probability is not between 0 and 1.05, something is wrong. Check the log sum."
        df["p"] = df["p"].apply(lambda x: min(x, 1))

        # Make sure all the bases are here
        assert len(df) == len(
            self.sequence
        ), "The number of bases in the sequence is not the same as the number of bases in the prediction."

        return df["p"].tolist()

    def __write_dms_to_file(self, sequence, signal):
        assert len(sequence) == len(
            signal
        ), "The length of the sequence is not the same as the length of the signal."

        with open(self.dms_file, "w") as f:
            for idx, (s, b) in enumerate(zip(signal, sequence)):
                if b in "AC":
                    f.write(f"{idx+1}\t{s}\n")

    def __make_temp_folder(self):
        """Remove content and make a new folder for temporary files."""
        path = 'temp'
        if os.path.exists(path):
            import shutil

            shutil.rmtree(path)
        os.makedirs(path)

    def __make_files(self, temp_prefix="temp"):
        this_file = lambda x: os.path.join('temp', x)
        self.pfs_file = this_file(f"{temp_prefix}.pfs")
        self.ct_file = this_file(f"{temp_prefix}.ct")
        self.dms_file = this_file(f"{temp_prefix}.shape")
        self.dot_file = this_file(f"{temp_prefix}_dot.txt")
        self.fasta_file = this_file(f"{temp_prefix}.fasta")
        self.prob_file = this_file(f"{temp_prefix}_prob.txt")

    def __create_fasta_file(self, reference, sequence):
        # push the ref into a temp file
        temp_fasta = open(self.fasta_file, "w")
        temp_fasta.write(">" + reference + "\n" + sequence)
        temp_fasta.close()

    def predictPairingProbability(self, sequence, dms=None, reference="reference"):
        self.sequence = sequence
        self.__make_temp_folder()
        self.__make_files()
        self.__create_fasta_file(reference, sequence)
        return self.predict_partition(dms=dms)

    def predictStructure(self, sequence, dms=None, shape=None):
        self.sequence = sequence
        self.__make_temp_folder()
        self.__make_files()
        self.__create_fasta_file("reference", sequence)
        envir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        import sys

        files = sys.path
        cmd = f"{os.path.join(rnastructure_path, 'Fold')} {self.fasta_file} {self.ct_file}"
        if type(dms) != type(None):
            assert len(sequence) == len(
                dms
            ), "The length of the sequence is not the same as the length of the signal."
            assert type(dms) in [
                list,
                tuple,
                np.ndarray,
            ], "The dms signal should be a list of floats."
            self.__write_dms_to_file(sequence, dms)
            cmd += " --dms " + self.dms_file
        if type(shape) != type(None):
            assert len(sequence) == len(
                shape
            ), "The length of the sequence is not the same as the length of the signal."
            assert type(shape) in [
                list,
                tuple,
                np.ndarray,
            ], "The shape signal should be a list of floats."
            self.__write_dms_to_file(sequence, shape)
            cmd += " --shape " + self.dms_file
        run_command(cmd)
        cmd = f"{os.path.join(rnastructure_path, 'ct2dot')} {self.ct_file} 0 {self.dot_file}"
        run_command(cmd)
        assert os.path.exists(
            self.dot_file
        ), 'The dot file was not created. Check RNAstructure installation. If you use a Mac, make sure to run `setup_env(RNASTRUCTURE_PATH="abs/path/to/RNAstructure/exe")`.'
        with open(self.dot_file, "r") as f:
            return f.readlines()[2].strip()