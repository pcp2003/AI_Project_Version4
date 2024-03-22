package breakout;

import utils.Commons;

public class Main {

    public static void main(String[] args) {

        GeneticAlgorithm ga = new GeneticAlgorithm();
        BreakoutBoard gaBB = new BreakoutBoard(ga.search(), false, Commons.SEED);
        gaBB.setSeed(Commons.SEED);
        gaBB.runSimulation();

//        new Breakout(ga.search(), Commons.SEED);


    }
}

