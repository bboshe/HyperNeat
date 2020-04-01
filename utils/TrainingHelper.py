5from IPython.display import clear_output
import pickle
import gzip
import utils.CSVReporter
import neat
import os


def train_continuously(evaluator, config, max_runs=10, location = "saves2/", fps=60):
    #if os.listdir(location):
    #    raise Exception("Directory is not empty!")
    org_max_stagnation = config.stagnation_config.max_stagnation

    for i in range(max_runs):
        clear_output(wait=True)
        pop = neat.Population(config)
        pop.add_reporter(utils.CSVReporter.CSVReporter(location + f"/lc_{i}.csv"))

        try:
            config.stagnation_config.max_stagnation = org_max_stagnation
            for j in range(1000000):
                gnome = pop.run(evaluator.eval_all_genomes, 1)

                if gnome.fitness < 1000:
                    config.stagnation_config.max_stagnation = int(org_max_stagnation * 0.5)
                elif gnome.fitness < 2000:
                    config.stagnation_config.max_stagnation = int(org_max_stagnation * 0.7)
                elif gnome.fitness < 3000:
                    config.stagnation_config.max_stagnation = int(org_max_stagnation * 1)
                else:
                    config.stagnation_config.max_stagnation = int(org_max_stagnation * 2)

        except neat.CompleteExtinctionException as e:
            gnome = pop.best_genome


        evaluator.create_video(gnome, config, fps=fps, fout=location + f"vis_{int(gnome.fitness)}_{i}.mp4")
        with gzip.open(location + f"cp_{i}.cp", 'w', compresslevel=5) as f:
            pickle.dump((gnome, config), f, protocol=pickle.HIGHEST_PROTOCOL)
