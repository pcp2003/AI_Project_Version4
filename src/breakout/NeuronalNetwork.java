package breakout;

import utils.Commons;
import utils.GameController;

public class NeuronalNetwork implements GameController, Comparable<NeuronalNetwork> {

    public int inputDim = Commons.BREAKOUT_STATE_SIZE;
    public int hiddenDim = Commons.BREAKOUT_HIDDENDIM_SIZE;
    public int outputDim = Commons.BREAKOUT_NUM_ACTIONS;
    private double[][] hiddenWeights;
    private double[] hiddenBiases;
    private double[][] outputWeights;
    private double[] outputBiases;

    public NeuronalNetwork(int inputDim, int hiddenDim, int outputDim) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        initializeParameters();
    }

    public NeuronalNetwork() {
        initializeParameters();
    }

    public NeuronalNetwork(double[] values) {
        fillParametersWithValues(values);
    }

    public NeuronalNetwork(int inputDim, int hiddenDim, int outputDim, double[] values) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        fillParametersWithValues(values);

    }

    public void fillParametersWithValues(double[] values) {

        // [w1,1; w1,2; w2,1; w2,2; B1; B2; w1,o; w2,o; Bo]
        hiddenWeights = new double[inputDim][hiddenDim];
        hiddenBiases = new double[hiddenDim];
        outputWeights = new double[hiddenDim][outputDim];
        outputBiases = new double[outputDim];

        int index = 0;

        for (int i = 0; i != inputDim; i++) {
            for (int j = 0; j != hiddenDim; j++) {
                hiddenWeights[i][j] = values[index++];
            }
        }

        for (int i = 0; i != hiddenDim; i++) {
            hiddenBiases[i] = values[index++];
        }

        for (int i = 0; i != hiddenDim; i++) {
            for (int j = 0; j != outputDim; j++) {
                outputWeights[i][j] = values[index++];

            }
        }

        for (int i = 0; i != outputDim; i++) {
            outputBiases[i] = values[index++];
        }
    }

    public void initializeParameters() {

        hiddenWeights = new double[inputDim][hiddenDim];
        hiddenBiases = new double[hiddenDim];
        outputWeights = new double[hiddenDim][outputDim];
        outputBiases = new double[outputDim];

        for (int i = 0; i != inputDim; i++) {
            for (int j = 0; j != hiddenDim; j++) {
                hiddenWeights[i][j] = Math.random() * 0.3;
            }
        }

        for (int i = 0; i != hiddenDim; i++) {
            hiddenBiases[i] = Math.random() * 0.3;
        }

        for (int i = 0; i != hiddenDim; i++) {
            for (int j = 0; j != outputDim; j++) {
                outputWeights[i][j] = Math.random() * 0.3;
            }
        }

        for (int i = 0; i != outputDim; i++) {
            outputBiases[i] = Math.random() * 0.3;
        }

    }

//    public void initializeParameters() {
//
//        hiddenWeights = new double[inputDim][hiddenDim];
//        hiddenBiases = new double[hiddenDim];
//        outputWeights = new double[hiddenDim][outputDim];
//        outputBiases = new double[outputDim];
//
//        // Inicialização de Xavier para os pesos da camada oculta
//        double lowerBoundHidden = -Math.sqrt(6.0 / (inputDim + hiddenDim));
//        double upperBoundHidden = Math.sqrt(6.0 / (inputDim + hiddenDim));
//
//        for (int i = 0; i < inputDim; i++) {
//            for (int j = 0; j < hiddenDim; j++) {
//                hiddenWeights[i][j] = lowerBoundHidden + Math.random() * (upperBoundHidden - lowerBoundHidden);
//            }
//        }
//
//        // Inicialização de Xavier para os pesos da camada de saída
//        double lowerBoundOutput = -Math.sqrt(6.0 / (hiddenDim + outputDim));
//        double upperBoundOutput = Math.sqrt(6.0 / (hiddenDim + outputDim));
//        for (int i = 0; i < hiddenDim; i++) {
//            for (int j = 0; j < outputDim; j++) {
//                outputWeights[i][j] = lowerBoundOutput + Math.random() * (upperBoundOutput - lowerBoundOutput);
//            }
//        }
//
//        // Inicializar vieses para 0 ou um pequeno valor aleatório próximo de 0
//        for (int i = 0; i < hiddenDim; i++) {
//            hiddenBiases[i] = 0; // Ou um valor pequeno aleatório se preferir
//        }
//        for (int i = 0; i < outputDim; i++) {
//            outputBiases[i] = 0; // Ou um valor pequeno aleatório se preferir
//        }
//    }

    public double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // next move e o foward da rede
    @Override
    public int nextMove(double[] currentState) {

        double[] fowardState = forward(currentState);

        System.out.println(" LEFT -->" + fowardState[0]);
        System.out.println(" RIGHT -->" + fowardState[1]);

        if (fowardState[0] > fowardState[1])
            return BreakoutBoard.LEFT;
        else if (fowardState[0] < fowardState[1])
            return BreakoutBoard.RIGHT;
        else
            return 0;
    }

    // Método para normalizar os dados de entrada
    private double[] normalizeInput(double[] inputValues) {

        double[] normalizedValues = new double[inputValues.length];
        double Xmin = 0; // Valor mínimo para cada dimensão da entrada
        double Xmax = 100; // Valor máximo para cada dimensão da entrada (ajuste conforme necessário)

        for (int i = 0; i < inputValues.length; i++) {
            normalizedValues[i] = (inputValues[i] - Xmin) / (Xmax - Xmin);
        }

        return normalizedValues;
    }

    public double[] forward(double[] inputValues) {

        double[] normalizedInputValues = normalizeInput(inputValues);

        double[] hiddenLayerOutput = new double[hiddenDim];
        double[] outputLayerOutput = new double[outputDim];

        for (int i = 0; i != hiddenDim; i++) {
            double res = 0;
            for (int j = 0; j != inputDim; j++) {
                res += hiddenWeights[j][i] * normalizedInputValues[j];
            }
            res += hiddenBiases[i];
            hiddenLayerOutput[i] = sigmoid(res);
        }


        for (int i = 0; i != outputDim; i++) {
            double res = 0;
            for (int j = 0; j != hiddenDim; j++) {
                res += outputWeights[j][i] * hiddenLayerOutput[j];
            }
            res += outputBiases[i];
            outputLayerOutput[i] = sigmoid(res);
        }

        return outputLayerOutput;
    }

    public double[] getNeuralNetwork() {

        double[] wheightsAndBiases = new double[(inputDim * hiddenDim) + hiddenDim + (hiddenDim * outputDim) + outputDim];

        int index = 0;

        for (int i = 0; i != inputDim; i++) {
            for (int j = 0; j != hiddenDim; j++) {
                wheightsAndBiases[index++] = hiddenWeights[i][j];
            }
        }

        for (int i = 0; i != hiddenDim; i++) {
            wheightsAndBiases[index++] = hiddenBiases[i];
        }

        for (int i = 0; i != hiddenDim; i++) {
            for (int j = 0; j != outputDim; j++) {
                wheightsAndBiases[index++] = outputWeights[i][j];

            }
        }

        for (int i = 0; i != outputDim; i++) {
            wheightsAndBiases[index++] = outputBiases[i];
        }

        return wheightsAndBiases;
    }

    @Override
    public String toString() {
        String result = "Neural Network: \nNumber of inputs: "
                + inputDim + "\n"
                + "Weights between input and hidden layer with " + hiddenDim + " neurons: \n";
        String hidden = "";
        for (int input = 0; input < inputDim; input++) {
            for (int i = 0; i < hiddenDim; i++) {
                hidden += " input" + input + "-hidden" + i + ": "
                        + hiddenWeights[input][i] + "\n";
            }
        }
        result += hidden;
        String biasHidden = "Hidden biases: \n";
        for (int i = 0; i < hiddenDim; i++) {
            biasHidden += " bias hidden" + i + ": " + hiddenBiases[i] + "\n";
        }
        result += biasHidden;
        String output = "Weights between hidden and output layer with "
                + outputDim + " neurons: \n";
        for (int hiddenw = 0; hiddenw < hiddenDim; hiddenw++) {
            for (int i = 0; i < outputDim; i++) {
                output += " hidden" + hiddenw + "-output" + i + ": "
                        + outputWeights[hiddenw][i] + "\n";
            }
        }
        result += output;
        String biasOutput = "Ouput biases: \n";
        for (int i = 0; i < outputDim; i++) {
            biasOutput += " bias ouput" + i + ": " + outputBiases[i] + "\n";
        }
        result += biasOutput;
        return result;
    }

    public int compareTo(NeuronalNetwork other) {

        BreakoutBoard otherBB = new BreakoutBoard(other, false, Commons.SEED);
        otherBB.setSeed(Commons.SEED);
        otherBB.runSimulation();

        BreakoutBoard thisBB = new BreakoutBoard(this, false, Commons.SEED);
        thisBB.setSeed(Commons.SEED);
        thisBB.runSimulation();

        return (int) (otherBB.getFitness() - thisBB.getFitness());
    }

}
