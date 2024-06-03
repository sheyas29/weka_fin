import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import com.formdev.flatlaf.FlatLightLaf;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.JTableHeader;
import java.awt.*;
import java.awt.geom.RoundRectangle2D;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

public class ChatbotApp extends JFrame {
    private JPanel chatPanel;
    private JScrollPane scrollPane;
    private JTextField inputField;
    private JButton sendButton;
    private J48 j48Classifier;
    private Instances data;
    private static final Color PRIMARY_COLOR = new Color(52, 152, 219);
    private static final Color SECONDARY_COLOR = new Color(44, 62, 80);
    private static final Color USER_MESSAGE_COLOR = new Color(135, 206, 250); // Light Blue
    private static final Color BOT_MESSAGE_COLOR = new Color(240, 240, 240); // Light Gray
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

        chatPanel = new JPanel();
        chatPanel.setLayout(new BoxLayout(chatPanel, BoxLayout.Y_AXIS));
        chatPanel.setBackground(Color.WHITE);

        scrollPane = new JScrollPane(chatPanel);
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
        addMessageToChat(userInput, true);

        String command = getClosestCommand(userInput);
        if (command == null) {
            addMessageToChat("Command not recognized. Please try again.", false);
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
                addMessageToChat("Invalid command.", false);
        }
    }

    private void addMessageToChat(String message, boolean isUser) {
        JPanel messagePanel = new JPanel();
        messagePanel.setLayout(new BoxLayout(messagePanel, BoxLayout.Y_AXIS));
        messagePanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        messagePanel.setBackground(isUser ? USER_MESSAGE_COLOR : BOT_MESSAGE_COLOR);

        JLabel messageLabel = new JLabel("<html><p style='width: 400px;'>" + message + "</p></html>");
        messageLabel.setFont(new Font("Segoe UI", Font.PLAIN, 14));
        messageLabel.setForeground(Color.BLACK);

        messagePanel.add(messageLabel);
        chatPanel.add(messagePanel);
        chatPanel.revalidate();
        scrollPane.getVerticalScrollBar().setValue(scrollPane.getVerticalScrollBar().getMaximum());
    }

    private void addTableToChat(String[][] data, String[] columnNames) {
        DefaultTableModel tableModel = new DefaultTableModel(data, columnNames);
        JTable table = new JTable(tableModel);
        styleTable(table);

        // Calculate the preferred height of the table
        int tableHeight = table.getRowHeight() * table.getRowCount() + table.getTableHeader().getPreferredSize().height;

        JScrollPane tableScrollPane = new JScrollPane(table);
        tableScrollPane.setPreferredSize(new Dimension(780, tableHeight));

        JPanel tablePanel = new JPanel(new BorderLayout());
        tablePanel.setBackground(BOT_MESSAGE_COLOR);
        tablePanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        tablePanel.add(tableScrollPane, BorderLayout.CENTER);

        chatPanel.add(tablePanel);
        chatPanel.revalidate();
        scrollPane.getVerticalScrollBar().setValue(scrollPane.getVerticalScrollBar().getMaximum());
    }

    private void styleTable(JTable table) {
        table.setFont(new Font("Segoe UI", Font.PLAIN, 14));
        table.setRowHeight(30);

        JTableHeader header = table.getTableHeader();
        header.setFont(new Font("Segoe UI", Font.BOLD, 14));
        header.setBackground(PRIMARY_COLOR);
        header.setForeground(Color.WHITE);

        DefaultTableCellRenderer centerRenderer = new DefaultTableCellRenderer();
        centerRenderer.setHorizontalAlignment(JLabel.CENTER);

        for (int i = 0; i < table.getColumnCount(); i++) {
            table.getColumnModel().getColumn(i).setCellRenderer(centerRenderer);
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
            for (int j = 0; j <= b.length(); j++) { // Fixed loop condition here
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
                addMessageToChat("Dataset loaded successfully.", false);
            } catch (Exception e) {
                addMessageToChat("Error loading dataset. Please check the path and format.", false);
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
            addMessageToChat("Error generating scatter plot.", false);
            e.printStackTrace();
        }
    }

    private void displayAllColumnStats(Instances data) {
        String[][] statsData = new String[data.numAttributes()][];
        String[] columnNames = { "Attribute", "Type", "Minimum", "Maximum", "Mean", "Std Dev" };
        for (int i = 0; i < data.numAttributes(); i++) {
            String[] statsRow = new String[6];
            statsRow[0] = data.attribute(i).name();
            if (data.attribute(i).isNumeric()) {
                double[] stats = calculateColumnStats(data, i);
                statsRow[1] = "Numeric";
                statsRow[2] = String.valueOf(stats[0]);
                statsRow[3] = String.valueOf(stats[1]);
                statsRow[4] = String.valueOf(stats[2]);
                statsRow[5] = String.valueOf(stats[3]);
            } else {
                statsRow[1] = "Nominal";
                statsRow[2] = "-";
                statsRow[3] = "-";
                statsRow[4] = "-";
                statsRow[5] = String.valueOf(data.attributeStats(i).nominalCounts.length);
            }
            statsData[i] = statsRow;
        }
        addTableToChat(statsData, columnNames);
    }

    private void compareTwoColumns(Instances data) {
        StringBuilder comparisonBuilder = new StringBuilder();
        comparisonBuilder.append("\n***** Column Indices and Names *****\n");
        for (int i = 0; i < data.numAttributes(); i++) {
            comparisonBuilder.append("Index ").append(i).append(": ").append(data.attribute(i).name()).append("\n");
        }
        int column1Index = Integer.parseInt(JOptionPane.showInputDialog(this,
                "Enter the index of the first column to compare:"));
        int column2Index = Integer.parseInt(JOptionPane.showInputDialog(this,
                "Enter the index of the second column to compare:"));
        if (!data.attribute(column1Index).isNumeric() || !data.attribute(column2Index).isNumeric()) {
            addMessageToChat("Error: Both columns must be numeric for comparison.", false);
            return;
        }
        double[] stats1 = calculateColumnStats(data, column1Index);
        double[] stats2 = calculateColumnStats(data, column2Index);
        String[][] comparisonData = {
                { "Minimum", String.valueOf(stats1[0]), String.valueOf(stats2[0]) },
                { "Maximum", String.valueOf(stats1[1]), String.valueOf(stats2[1]) },
                { "Mean", String.valueOf(stats1[2]), String.valueOf(stats2[2]) },
                { "Std Dev", String.valueOf(stats1[3]), String.valueOf(stats2[3]) }
        };
        String[] columnNames = { "Statistic", data.attribute(column1Index).name(),
                data.attribute(column2Index).name() };
        addTableToChat(comparisonData, columnNames);
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
            addMessageToChat("Column Names for Dataset " + (i + 1) + ":", false);
            String[][] columnData = new String[datasets.get(i).numAttributes()][2];
            for (int j = 0; j < datasets.get(i).numAttributes(); j++) {
                columnData[j][0] = String.valueOf(j);
                columnData[j][1] = datasets.get(i).attribute(j).name();
            }
            String[] columnNames = { "Index", "Name" };
            addTableToChat(columnData, columnNames);
        }

        int numOfColumns = Integer
                .parseInt(JOptionPane.showInputDialog(this, "How many column comparisons would you like to do?"));
        for (int i = 0; i < numOfColumns; i++) {
            addMessageToChat("Comparison Pair " + (i + 1) + ":", false);

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

            String[][] comparisonData = new String[numOfColumnsToCompare][5];
            for (int j = 0; j < numOfColumnsToCompare; j++) {
                if (datasetIndices.get(j) < 0 || datasetIndices.get(j) >= numDatasets || columnIndices.get(j) < 0
                        || columnIndices.get(j) >= datasets.get(datasetIndices.get(j)).numAttributes()) {
                    addMessageToChat("Error: Invalid dataset or column index.", false);
                    return;
                }

                if (!datasets.get(datasetIndices.get(j)).attribute(columnIndices.get(j)).isNumeric()) {
                    addMessageToChat("Error: All columns must be numeric for comparison.", false);
                    return;
                }

                double[] stats = calculateColumnStats(datasets.get(datasetIndices.get(j)), columnIndices.get(j));
                comparisonData[j][0] = datasets.get(datasetIndices.get(j)).attribute(columnIndices.get(j)).name()
                        + " (Dataset " + (datasetIndices.get(j) + 1) + ")";
                comparisonData[j][1] = String.valueOf(stats[0]);
                comparisonData[j][2] = String.valueOf(stats[1]);
                comparisonData[j][3] = String.valueOf(stats[2]);
                comparisonData[j][4] = String.valueOf(stats[3]);
            }
            String[] columnNames = { "Column", "Minimum", "Maximum", "Mean", "Std Dev" };
            addTableToChat(comparisonData, columnNames);
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
                StringBuilder predictionsBuilder = new StringBuilder();
                predictionsBuilder.append("Predictions for the new unlabelled dataset:\n");
                for (int i = 0; i < newData.numInstances(); i++) {
                    double predictedClass = j48Classifier.classifyInstance(newData.instance(i));
                    predictionsBuilder.append("Instance ").append(i + 1).append(": Predicted class - ")
                            .append(data.classAttribute().value((int) predictedClass)).append("\n");
                }
                Evaluation eval = new Evaluation(data);
                eval.evaluateModel(j48Classifier, newData);
                int response = JOptionPane.showConfirmDialog(this, "Do you want to see evaluation metrics?",
                        "Evaluation Metrics", JOptionPane.YES_NO_OPTION);
                if (response == JOptionPane.YES_OPTION) {
                    predictionsBuilder.append(eval.toSummaryString()).append("\n");
                    predictionsBuilder.append(eval.toMatrixString()).append("\n");
                    predictionsBuilder.append(eval.toClassDetailsString()).append("\n");
                }
                addMessageToChat(predictionsBuilder.toString(), false);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new ChatbotApp().setVisible(true));
    }
}
