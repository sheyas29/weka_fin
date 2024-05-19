import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.util.ArrayList;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.*;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;
import weka.core.Instance;

public class ChatbotApp extends JFrame {
    private JTextArea chatArea;
    private JTextField inputField;
    private JButton sendButton;
    private J48 j48Classifier;
    private Instances data;

    private static final String[] COMMANDS = {
            "visualize dataset",
            "statistics for all columns",
            "compare two columns",
            "predict",
            "compare datasets"
    };

    public ChatbotApp() {

        setTitle("Data Analysis Chatbot");
        setSize(600, 400);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        chatArea = new JTextArea();
        chatArea.setEditable(false);
        add(new JScrollPane(chatArea), BorderLayout.CENTER);

        JPanel inputPanel = new JPanel(new BorderLayout());
        inputField = new JTextField();
        sendButton = new JButton("Send");

        inputPanel.add(inputField, BorderLayout.CENTER);
        inputPanel.add(sendButton, BorderLayout.EAST);
        add(inputPanel, BorderLayout.SOUTH);

        sendButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                processUserInput();
            }
        });

        inputField.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                processUserInput();
            }
        });

        loadDataset(); // Load the dataset when the app starts
        if (data != null) {
            j48Classifier = trainJ48Classifier(data); // Train only if dataset loaded successfully
        }
    }

    private void processUserInput() {
        String userInput = inputField.getText().trim().toLowerCase();
        inputField.setText("");

        chatArea.append("You: " + userInput + "\n");

        String command = getClosestCommand(userInput);

        if (command == null) {
            chatArea.append("Bot: Command not recognized. Please try again.\n");
            return;
        }

        switch (command) {
            case "visualize dataset":
                visualizeDataset(data);
                break;
            case "statistics for all columns":
                displayAllColumnStats(data);
                break;
            case "compare two columns":
                compareTwoColumns(data);
                break;
            case "predict":
                predictWithJ48(data);
                break;
            case "compare datasets":
                compareDatasets();
                break;
            default:
                chatArea.append("Bot: Invalid command.\n");
        }
    }

    private String getClosestCommand(String userInput) {
        int minDistance = Integer.MAX_VALUE;
        String closestCommand = null;

        for (String command : COMMANDS) {
            int distance = levenshteinDistance(userInput, command);
            if (distance < minDistance) {
                minDistance = distance;
                closestCommand = command;
            }
        }

        return minDistance <= 3 ? closestCommand : null;
    }

    private int levenshteinDistance(String a, String b) {
        int[][] dp = new int[a.length() + 1][b.length() + 1];

        for (int i = 0; i <= a.length(); i++) {
            for (int j = 0; j <= b.length(); j++) {
                if (i == 0) {
                    dp[i][j] = j;
                } else if (j == 0) {
                    dp[i][j] = i;
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1] + (a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1),
                            Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1));
                }
            }
        }

        return dp[a.length()][b.length()];
    }

    private void loadDataset() {
        String datasetPath = JOptionPane.showInputDialog(this,
                "Enter the path to your dataset (CSV format):");
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(datasetPath)); // Load from user-provided path
            data = loader.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1); // Set class index (if not already set)
            }
            chatArea.append("Bot: Dataset loaded successfully.\n");
        } catch (Exception e) {
            chatArea.append("Bot: Error loading dataset. Please check the path and format.\n");
            e.printStackTrace();
        }
    }

    private J48 trainJ48Classifier(Instances data) {
        try {
            J48 j48 = new J48();
            j48.buildClassifier(data);
            return j48;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private void visualizeDataset(Instances data) {
        JFrame frame = new JFrame("Dataset Visualization");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(800, 600);

        visualizeScatterPlot(data);

        frame.setVisible(true);
    }

    private void visualizeScatterPlot(Instances data) {
        int attributeIndex1 = Integer.parseInt(JOptionPane.showInputDialog(this,
                "Enter the index of the first numeric attribute to visualize:"));
        int attributeIndex2 = Integer.parseInt(JOptionPane.showInputDialog(this,
                "Enter the index of the second numeric attribute to visualize:"));

        try {
            // Create a VisualizePanel
            VisualizePanel visualizePanel = new VisualizePanel();

            // Prepare data for plotting
            Instances plotData = new Instances(data);
            plotData.setClassIndex(attributeIndex1); // Set the class index

            // PlotData2D object to hold plot data
            PlotData2D plotData2D = new PlotData2D(plotData);
            plotData2D.setPlotName("Scatter Plot");
            plotData2D.addInstanceNumberAttribute();

            // Add the plot to the panel
            visualizePanel.addPlot(plotData2D);

            // Set up the frame
            JFrame frame = new JFrame("Scatter Plot");
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.setSize(800, 600);
            frame.getContentPane().add(visualizePanel);
            frame.setVisible(true);
        } catch (Exception e) {
            chatArea.append("Error generating scatter plot.\n");
            e.printStackTrace();
        }
    }

    private void displayAllColumnStats(Instances data) {
        // Existing statistics display logic
        for (int i = 0; i < data.numAttributes(); i++) {
            chatArea.append("\n***** Attribute " + (i + 1) + ": " + data.attribute(i).name() + " *****\n");
            if (data.attribute(i).isNumeric()) {
                double[] stats = calculateColumnStats(data, i);
                chatArea.append("Type: Numeric\n");
                chatArea.append("Minimum: " + stats[0] + "\n");
                chatArea.append("Maximum: " + stats[1] + "\n");
                chatArea.append("Mean: " + stats[2] + "\n");
                chatArea.append("Standard Deviation: " + stats[3] + "\n");
            } else {
                chatArea.append("Type: Nominal\n");
                chatArea.append("Distinct Values: " + data.attributeStats(i).nominalCounts.length + "\n");
            }
        }
    }

    private void compareTwoColumns(Instances data) {
        // Existing column comparison logic
        chatArea.append("\n***** Column Indices and Names *****\n");
        for (int i = 0; i < data.numAttributes(); i++) {
            chatArea.append("Index " + i + ": " + data.attribute(i).name() + "\n");
        }

        int column1Index = Integer
                .parseInt(JOptionPane.showInputDialog(this, "Enter the index of the first column to compare:"));
        int column2Index = Integer
                .parseInt(JOptionPane.showInputDialog(this, "Enter the index of the second column to compare:"));

        if (!data.attribute(column1Index).isNumeric() || !data.attribute(column2Index).isNumeric()) {
            chatArea.append("Error: Both columns must be numeric for comparison.\n");
            return;
        }

        double[] stats1 = calculateColumnStats(data, column1Index);
        double[] stats2 = calculateColumnStats(data, column2Index);

        chatArea.append("\n***** Column Comparison *****\n");
        chatArea.append(String.format("Minimum:  %s = %.1f, %s = %.1f\n", data.attribute(column1Index).name(),
                stats1[0], data.attribute(column2Index).name(), stats2[0]));
        chatArea.append(String.format("Maximum:  %s = %.1f, %s = %.1f\n", data.attribute(column1Index).name(),
                stats1[1], data.attribute(column2Index).name(), stats2[1]));
        chatArea.append(String.format("Mean:     %s = %.3f, %s = %.3f\n", data.attribute(column1Index).name(),
                stats1[2], data.attribute(column2Index).name(), stats2[2]));
        chatArea.append(String.format("Std Dev:  %s = %.3f, %s = %.3f\n", data.attribute(column1Index).name(),
                stats1[3], data.attribute(column2Index).name(), stats2[3]));
    }

    private double[] calculateColumnStats(Instances data, int columnIndex) {
        if (data.attribute(columnIndex).isNumeric()) {
            double min = data.attributeStats(columnIndex).numericStats.min;
            double max = data.attributeStats(columnIndex).numericStats.max;
            double mean = data.attributeStats(columnIndex).numericStats.mean;
            double stdDev = data.attributeStats(columnIndex).numericStats.stdDev;
            return new double[] { min, max, mean, stdDev };
        } else {
            return null;
        }
    }

    private void compareDatasets() {
        // Existing dataset comparison logic with JOptionPane dialogs for input
        int numDatasets = Integer
                .parseInt(JOptionPane.showInputDialog(this, "Enter the number of datasets to compare:"));

        java.util.List<Instances> datasets = new ArrayList<>();
        for (int i = 0; i < numDatasets; i++) {
            String filePath = JOptionPane.showInputDialog(this,
                    "Enter the path to dataset " + (i + 1) + " (CSV format):");

            try {
                CSVLoader loader = new CSVLoader();
                loader.setSource(new File(filePath));
                Instances dataset = loader.getDataSet();
                datasets.add(dataset);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        for (int i = 0; i < datasets.size(); i++) {
            chatArea.append("\nColumn Names for Dataset " + (i + 1) + ":\n");
            for (int j = 0; j < datasets.get(i).numAttributes(); j++) {
                chatArea.append("Index " + j + ": " + datasets.get(i).attribute(j).name() + "\n");
            }
        }

        int numOfColumns = Integer
                .parseInt(JOptionPane.showInputDialog(this, "How many column comparisons would you like to do?"));

        for (int i = 0; i < numOfColumns; i++) {
            chatArea.append("\nComparison Pair " + (i + 1) + ":\n");

            java.util.List<Integer> datasetIndices = new ArrayList<>();
            java.util.List<Integer> columnIndices = new ArrayList<>();

            int numOfColumnsToCompare = Integer.parseInt(
                    JOptionPane.showInputDialog(this, "How many columns would you like to compare in this pair?"));

            for (int j = 0; j < numOfColumnsToCompare; j++) {
                int datasetIndex = Integer.parseInt(JOptionPane.showInputDialog(this,
                        "Enter the dataset index (1-" + numDatasets + ") for the column:")) - 1;
                datasetIndices.add(datasetIndex);

                int columnIndex = Integer.parseInt(JOptionPane.showInputDialog(this,
                        "Enter the column index in dataset " + (datasetIndex + 1) + ":"));
                columnIndices.add(columnIndex);
            }

            for (int j = 0; j < numOfColumnsToCompare; j++) {
                if (datasetIndices.get(j) < 0 || datasetIndices.get(j) >= numDatasets || columnIndices.get(j) < 0
                        || columnIndices.get(j) >= datasets.get(datasetIndices.get(j)).numAttributes()) {
                    chatArea.append("Error: Invalid dataset or column index.\n");
                    return;
                }

                if (!datasets.get(datasetIndices.get(j)).attribute(columnIndices.get(j)).isNumeric()) {
                    chatArea.append("Error: All columns must be numeric for comparison.\n");
                    return;
                }
            }

            for (int j = 0; j < numOfColumnsToCompare; j++) {
                double[] stats = calculateColumnStats(datasets.get(datasetIndices.get(j)), columnIndices.get(j));

                chatArea.append("\nColumn " + (j + 1) + ": "
                        + datasets.get(datasetIndices.get(j)).attribute(columnIndices.get(j)).name() + " (Dataset "
                        + (datasetIndices.get(j) + 1) + ")\n");
                chatArea.append("Minimum: " + stats[0] + "\n");
                chatArea.append("Maximum: " + stats[1] + "\n");
                chatArea.append("Mean: " + stats[2] + "\n");
                chatArea.append("Standard Deviation: " + stats[3] + "\n");
            }
        }
    }

    private void predictWithJ48(Instances data) {
        // Existing prediction logic with JOptionPane dialogs for input
        String datasetPath = JOptionPane.showInputDialog(this,
                "Enter the path to the new unlabelled dataset (in CSV format):");

        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(datasetPath));
            Instances newData = loader.getDataSet();

            if (newData.classIndex() == -1 && newData.numAttributes() > 0) {
                newData.setClassIndex(newData.numAttributes() - 1);
            }

            chatArea.append("Predictions for the new unlabelled dataset:\n");
            for (int i = 0; i < newData.numInstances(); i++) {
                double predictedClass = j48Classifier.classifyInstance(newData.instance(i));
                chatArea.append("Instance " + (i + 1) + ": Predicted class - "
                        + data.classAttribute().value((int) predictedClass) + "\n");
            }

            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(j48Classifier, newData);

            int response = JOptionPane.showConfirmDialog(this, "Do you want to see evaluation metrics?",
                    "Evaluation Metrics", JOptionPane.YES_NO_OPTION);
            if (response == JOptionPane.YES_OPTION) {
                chatArea.append(eval.toSummaryString() + "\n");
                chatArea.append(eval.toMatrixString() + "\n");
                chatArea.append(eval.toClassDetailsString() + "\n");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new ChatbotApp().setVisible(true);
            }
        });
    }
}
