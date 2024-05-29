import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import com.formdev.flatlaf.FlatLightLaf;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import com.privatejgoodies.forms.factories.CC;
import com.privatejgoodies.forms.layout.FormLayout;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.RoundRectangle2D;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

public class ChatbotApp extends JFrame {
    private JTextArea chatArea;
    private JTextField inputField;
    private JButton sendButton;
    private J48 j48Classifier;
    private Instances data;
    private static final Color PRIMARY_COLOR = new Color(52, 152, 219);
    private static final Color SECONDARY_COLOR = new Color(44, 62, 80);
    private static final String[] COMMANDS = {
            "visualize dataset",
            "statistics for all columns",
            "compare two columns",
            "predict",
            "compare datasets"
    };

    public class RoundedButton extends JButton {
        private final int radius;

        public RoundedButton(String text, int radius) {
            super(text);
            this.radius = radius;
            setContentAreaFilled(false);
            setFocusPainted(false);
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            if (getModel().isArmed()) {
                g2.setColor(getBackground().darker());
            } else if (getModel().isRollover()) {
                g2.setColor(getBackground().brighter());
            } else {
                g2.setColor(getBackground());
            }
            g2.fillRoundRect(0, 0, getWidth() - 1, getHeight() - 1, radius, radius);
            super.paintComponent(g);
            g2.dispose();
        }

        @Override
        public boolean contains(int x, int y) {
            return new RoundRectangle2D.Float(0, 0, getWidth() - 1, getHeight() - 1, radius, radius).contains(x, y);
        }
    }

    public ChatbotApp() {
        setupLookAndFeel();
        initializeComponents();
        loadDataset();
        if (data != null) {
            j48Classifier = trainJ48Classifier(data);
        }
    }

    private void setupLookAndFeel() {
        try {
            UIManager.setLookAndFeel(new FlatLightLaf());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initializeComponents() {
        setTitle("Data Analysis Chatbot");
        setSize(800, 600);
        setLocationRelativeTo(null);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        chatArea = new JTextArea();
        chatArea.setFont(new Font("Segoe UI", Font.PLAIN, 16));
        chatArea.setLineWrap(true);
        chatArea.setWrapStyleWord(true);

        JScrollPane scrollPane = new JScrollPane(chatArea);
        scrollPane.setBorder(BorderFactory.createEmptyBorder());
        add(scrollPane, BorderLayout.CENTER);

        JPanel inputPanel = createInputPanel();
        add(inputPanel, BorderLayout.SOUTH);

        sendButton.addActionListener(e -> processUserInput());
        inputField.addActionListener(e -> processUserInput());
    }

    private JPanel createInputPanel() {
        JPanel inputPanel = new JPanel(new GridBagLayout());
        inputPanel.setBackground(Color.WHITE);
        inputPanel.setBorder(BorderFactory.createEmptyBorder(15, 20, 15, 20));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(5, 5, 5, 5);

        JLabel inputLabel = new JLabel("Enter your command:");
        inputLabel.setFont(new Font("Segoe UI", Font.BOLD, 14));
        inputLabel.setForeground(SECONDARY_COLOR);
        gbc.gridx = 0;
        gbc.gridy = 0;
        inputPanel.add(inputLabel, gbc);

        inputField = new JTextField();
        inputField.setFont(new Font("Segoe UI", Font.PLAIN, 14));
        inputField.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(0, 0, 2, 0, PRIMARY_COLOR),
                BorderFactory.createEmptyBorder(5, 10, 5, 10)));
        gbc.gridx = 1;
        gbc.gridy = 0;
        gbc.weightx = 1.0;
        inputPanel.add(inputField, gbc);

        sendButton = new RoundedButton("Send", 25);
        sendButton.setBackground(PRIMARY_COLOR);
        sendButton.setForeground(Color.WHITE);
        sendButton.setFont(new Font("Segoe UI", Font.BOLD, 14));
        gbc.gridx = 2;
        gbc.gridy = 0;
        inputPanel.add(sendButton, gbc);

        return inputPanel;
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
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Select Dataset");
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            try {
                CSVLoader loader = new CSVLoader();
                loader.setSource(selectedFile);
                data = loader.getDataSet();
                if (data.classIndex() == -1) {
                    data.setClassIndex(data.numAttributes() - 1);
                }
                chatArea.append("Bot: Dataset loaded successfully.\n");
            } catch (Exception e) {
                chatArea.append("Bot: Error loading dataset. Please check the path and format.\n");
                e.printStackTrace();
            }
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
        int attributeIndex1 = Integer.parseInt(JOptionPane.showInputDialog(this,
                "Enter the index of the first numeric attribute to visualize:"));
        int attributeIndex2 = Integer.parseInt(JOptionPane.showInputDialog(this,
                "Enter the index of the second numeric attribute to visualize:"));
        try {
            PlotData2D plotData = new PlotData2D(data);
            plotData.setPlotName("Scatter Plot");
            plotData.addInstanceNumberAttribute();
            plotData.setXindex(attributeIndex1);
            plotData.setYindex(attributeIndex2);
            VisualizePanel visualizePanel = new VisualizePanel();
            visualizePanel.setName("Scatter Plot");
            visualizePanel.addPlot(plotData);
            JFrame frame = new JFrame("Scatter Plot");
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.setSize(800, 600);
            frame.setLayout(new BorderLayout());
            frame.add(visualizePanel, BorderLayout.CENTER);
            frame.setVisible(true);
        } catch (Exception e) {
            chatArea.append("Error generating scatter plot.\n");
            e.printStackTrace();
        }
    }

    private void displayAllColumnStats(Instances data) {
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
        chatArea.append("\n***** Column Indices and Names *****\n");
        for (int i = 0; i < data.numAttributes(); i++) {
            chatArea.append("Index " + i + ": " + data.attribute(i).name() + "\n");
        }
        int column1Index = Integer.parseInt(JOptionPane.showInputDialog(this,
                "Enter the index of the first column to compare:"));
        int column2Index = Integer.parseInt(JOptionPane.showInputDialog(this,
                "Enter the index of the second column to compare:"));
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
        int numDatasets = Integer
                .parseInt(JOptionPane.showInputDialog(this, "Enter the number of datasets to compare:"));
        List<Instances> datasets = new ArrayList<>();
        for (int i = 0; i < numDatasets; i++) {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setDialogTitle("Select Dataset " + (i + 1));
            int result = fileChooser.showOpenDialog(this);
            if (result == JFileChooser.APPROVE_OPTION) {
                File selectedFile = fileChooser.getSelectedFile();
                try {
                    CSVLoader loader = new CSVLoader();
                    loader.setSource(selectedFile);
                    Instances dataset = loader.getDataSet();
                    datasets.add(dataset);
                } catch (Exception e) {
                    e.printStackTrace();
                }
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
            List<Integer> datasetIndices = new ArrayList<>();
            List<Integer> columnIndices = new ArrayList<>();
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
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Select Unlabelled Dataset");
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            try {
                CSVLoader loader = new CSVLoader();
                loader.setSource(selectedFile);
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
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new ChatbotApp().setVisible(true));
    }
}
