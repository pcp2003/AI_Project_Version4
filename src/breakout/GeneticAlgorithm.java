package breakout;

import utils.Commons;

import java.util.Arrays;

public class GeneticAlgorithm {

    private final int POPULATION_SIZE = 10;
    private final int NUM_GENERATIONS = 10;
    private final double MUTATION_RATE = 0.25;
    private final double SELECTION_PERCENTAGE = 0.3;

    private final int NeuralNetworkValuesSize = (Commons.BREAKOUT_STATE_SIZE * Commons.BREAKOUT_HIDDENDIM_SIZE) + Commons.BREAKOUT_HIDDENDIM_SIZE + (Commons.BREAKOUT_HIDDENDIM_SIZE * Commons.BREAKOUT_NUM_ACTIONS) + Commons.BREAKOUT_NUM_ACTIONS;

    private NeuronalNetwork[] population = new NeuronalNetwork[POPULATION_SIZE];

    public GeneticAlgorithm() {
        generatePopulation();
    }

    public NeuronalNetwork search() {

        for (int i = 0; i < NUM_GENERATIONS; i++) {

            Arrays.sort(population);

//            System.out.println("Generation " + i + ":\n" + population[0]);

            BreakoutBoard bestFitness = new BreakoutBoard(population[0], false, Commons.SEED);
            bestFitness.setSeed(Commons.SEED);
            bestFitness.runSimulation();
//
//            System.out.println("Generation " + i + " Best Fitness " + bestFitness.getFitness());

            NeuronalNetwork[] newGeneration = new NeuronalNetwork[POPULATION_SIZE];

            for (int j = 0; j < POPULATION_SIZE; j += 2) {
                NeuronalNetwork parent1 = selectParent();
                NeuronalNetwork parent2 = selectParent();
                NeuronalNetwork[] children = crossover(parent1, parent2);

                newGeneration[j] = mutate(children[0]);
                if (j + 1 < POPULATION_SIZE) {
                    newGeneration[j + 1] = mutate(children[1]);
                }
            }
            createNewPopulation(newGeneration);
        }
        return population[0];
    }

    //SELECTION_PERCENTAGE of the best children +
    //(1-SELECTION_PERCENTAGE) of the best from the previous population

    // Ex: Deixo os 25 melhores da populacao anterior, e substituo os 25 piores da populacao anterior pelo 25 melhores da nova geracao
    private void createNewPopulation(NeuronalNetwork[] newgeneration) {

        Arrays.sort(newgeneration);
        int cutoff = (int) (POPULATION_SIZE * SELECTION_PERCENTAGE);

        int worstIndividuals = POPULATION_SIZE - cutoff;

        for (int i = 0; i != cutoff; i++) {
            population[i + worstIndividuals] = newgeneration[i];
        }

    }

    // mutate one gene with MUTATION_RATE chance
    private NeuronalNetwork mutate(NeuronalNetwork child) {

        double random = Math.random();

        if (random <= MUTATION_RATE) {

            double[] childNewPos = child.getNeuralNetwork();
            childNewPos[(int) (Math.random() * NeuralNetworkValuesSize)] = (Math.random() * 1.0);
            return new NeuronalNetwork(childNewPos);
        }

        return child;
    }

    // one-point crossover
    // Estou dividindo o array no meio
    private NeuronalNetwork[] crossover(NeuronalNetwork parent1, NeuronalNetwork parent2) {

        int numberOfChildren = 2;
//        int random = (int) (Math.random() * NeuralNetworkValuesSize);
        int random = (NeuralNetworkValuesSize / 2);

        NeuronalNetwork[] children = new NeuronalNetwork[numberOfChildren];

        double[] child1 = new double[NeuralNetworkValuesSize];
        double[] child2 = new double[NeuralNetworkValuesSize];
        double[] parent1_positions = parent1.getNeuralNetwork();
        double[] parent2_positions = parent2.getNeuralNetwork();

        for (int j = 0; j != random; j++) {
            child1[j] = parent1_positions[j];
            child2[j] = parent2_positions[j];
        }
        for (int k = random; k != NeuralNetworkValuesSize; k++) {
            child1[k] = parent2_positions[k];
            child2[k] = parent1_positions[k];
        }
        children[0] = new NeuronalNetwork(child1);
        children[1] = new NeuronalNetwork(child2);

        return children;
    }

    // Usa apenas os 40 melhores da populacao para reproduzir filhos
    private NeuronalNetwork selectParent() {

        NeuronalNetwork one = population[(int) (Math.random() * 4)];
        NeuronalNetwork two = population[(int) (Math.random() * 4)];

        if (one.compareTo(two) >= 0)
            return two;
        else
            return one;

    }

    // generate random population
    private void generatePopulation() {
        for (int i = 0; i < population.length; i++) {
            population[i] = new NeuronalNetwork();
        }
    }

}

