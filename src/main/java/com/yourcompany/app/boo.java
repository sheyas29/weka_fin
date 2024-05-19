package com.yourcompany.app;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Scanner;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;

import javax.swing.*;
import java.awt.*;

import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;

public class boo {
    public static void main(String[] args) {
        try {
            // Load and prepare dataset (same as before)
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("D:/csv/Iris.csv"));
            Instances data = loader.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            J48 j48Classifier = trainJ48Classifier(data);

            Scanner scanner = new Scanner(System.in);
            boolean continueAnalysis = true;

            while (continueAnalysis) {
                System.out.println("\n***** Data Analysis *****");
                System.out.println("1. Visualize Dataset");
                System.out.println("2. Statistics for All Columns");
                System.out.println("3. Compare Two Columns");
                System.out.println("4. Predict");
                System.out.println("5. Compare Datasets"); // New option
                System.out.println("6. Exit");
                System.out.print("Enter your choice: ");

                int choice;
                try { // Input validation for choice
                    choice = scanner.nextInt();
                } catch (InputMismatchException e) {
                    System.out.println("Invalid input. Please enter a number.");
                    scanner.next(); // Clear invalid input
                    continue; // Continue to next iteration of loop
                }

                switch (choice) {
                    case 1:
                        visualizeDataset(data);
                        break;
                    case 2:
                        displayAllColumnStats(data);
                        break;
                    case 3:
                        compareTwoColumns(data, scanner);
                        break;
                    case 4:
                        predictWithJ48(data, scanner, j48Classifier);
                        break;
                    case 5:
                        compareDatasets(scanner);
                        break;
                    case 6:
                        continueAnalysis = false;
                        break;
                    default:
                        System.out.println("Invalid choice.");
                }
            }
            System.out.println("Exiting analysis...");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static J48 trainJ48Classifier(Instances data) {
        try {
            J48 j48 = new J48();
            j48.buildClassifier(data);
            return j48;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static void visualizeDataset(Instances data) {
        JFrame frame = new JFrame("Dataset Visualization");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(800, 600);

        JPanel panel = new JPanel();
        frame.add(panel);
        panel.setLayout(new BorderLayout());

        Scanner scanner = new Scanner(System.in);
        System.out.println("Choose visualization type:");
        System.out.println("1. Histogram");
        System.out.println("2. Scatter Plot");
        System.out.print("Enter your choice: ");
        int choice = scanner.nextInt();

        switch (choice) {
            case 1:
                visualizeHistogram(data);
                break;
            case 2:
                visualizeScatterPlot(data);
                break;
            default:
                System.out.println("Invalid choice. Exiting...");
                return;
        }

        frame.setVisible(true);
    }

    private static void visualizeHistogram(Instances data) {
        JFrame frame = new JFrame("Dataset Visualization");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(800, 600);

        JPanel panel = new JPanel();
        frame.add(panel);
        panel.setLayout(new BorderLayout());

        // Ask user for attribute index
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter the index of the numeric attribute to visualize: ");
        int attributeIndex = scanner.nextInt();

        // Create histogram dataset
        double[] values = data.attributeToDoubleArray(attributeIndex);
        HistogramDataset dataset = new HistogramDataset();
        dataset.addSeries("Histogram", values, 10); // Number of bins

        // Create histogram chart
        JFreeChart chart = ChartFactory.createHistogram(
                "Histogram", // Title
                "Values", // X-axis label
                "Frequency", // Y-axis label
                dataset);

        // Customize chart appearance
        chart.setBackgroundPaint(Color.WHITE);
        chart.getTitle().setPaint(Color.DARK_GRAY);

        // Customize plot appearance
        ((XYPlot) chart.getPlot()).setDomainGridlinePaint(Color.DARK_GRAY);
        ((XYPlot) chart.getPlot()).setRangeGridlinePaint(Color.DARK_GRAY);

        // Display chart in a ChartPanel
        ChartPanel chartPanel = new ChartPanel(chart);
        panel.add(chartPanel, BorderLayout.CENTER);

        frame.setVisible(true);
    }

    private static void visualizeScatterPlot(Instances data) {
        // Ask user for attribute indices
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter the index of the first numeric attribute to visualize: ");
        int attributeIndex1 = scanner.nextInt();
        System.out.print("Enter the index of the second numeric attribute to visualize: ");
        int attributeIndex2 = scanner.nextInt();

        // Create scatter plot dataset
        XYSeriesCollection dataset = new XYSeriesCollection();
        XYSeries series = new XYSeries("Scatter Plot");
        for (Instance instance : data) {
            double x = instance.value(attributeIndex1);
            double y = instance.value(attributeIndex2);
            series.add(x, y);
        }
        dataset.addSeries(series);

        // Create scatter plot chart
        JFreeChart chart = ChartFactory.createScatterPlot(
                "Scatter Plot", // Title
                data.attribute(attributeIndex1).name(), // X-axis label
                data.attribute(attributeIndex2).name(), // Y-axis label
                dataset);

        // Customize chart appearance
        chart.setBackgroundPaint(Color.WHITE);
        chart.getTitle().setPaint(Color.DARK_GRAY);

        // Display chart in a ChartPanel
        ChartPanel chartPanel = new ChartPanel(chart);
        JFrame frame = new JFrame("Scatter Plot");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.getContentPane().add(chartPanel);
        frame.pack();
        frame.setVisible(true);
    }

    private static void displayAllColumnStats(Instances data) {
        for (int i = 0; i < data.numAttributes(); i++) {
            System.out.println("\n***** Attribute " + (i + 1) + ": " + data.attribute(i).name() + " *****");
            if (data.attribute(i).isNumeric()) {
                double[] stats = calculateColumnStats(data, i); // Calculate stats
                System.out.println("Type: Numeric");
                System.out.println("Minimum: " + stats[0]);
                System.out.println("Maximum: " + stats[1]);
                System.out.println("Mean: " + stats[2]);
                System.out.println("Standard Deviation: " + stats[3]);
            } else {
                System.out.println("Type: Nominal");
                System.out.println("Distinct Values: " + data.attributeStats(i).nominalCounts.length);
            }
        }
    }

    private static void compareTwoColumns(Instances data, Scanner scanner) {
        System.out.println("\n***** Column Indices and Names *****");
        for (int i = 0; i < data.numAttributes(); i++) {
            System.out.println("Index " + i + ": " + data.attribute(i).name());
        }
        System.out.print("\nEnter the index of the first column to compare: ");
        int column1Index = scanner.nextInt();
        System.out.print("Enter the index of the second column to compare: ");
        int column2Index = scanner.nextInt();
        if (!data.attribute(column1Index).isNumeric() || !data.attribute(column2Index).isNumeric()) {
            System.out.println("Error: Both columns must be numeric for comparison.");
            return;
        }
        double[] stats1 = calculateColumnStats(data, column1Index);
        double[] stats2 = calculateColumnStats(data, column2Index);
        System.out.println("\n***** Column Comparison *****");
        System.out.printf("Minimum:  %s = %.1f, %s = %.1f\n",
                data.attribute(column1Index).name(), stats1[0],
                data.attribute(column2Index).name(), stats2[0]);
        System.out.printf("Maximum:  %s = %.1f, %s = %.1f\n",
                data.attribute(column1Index).name(), stats1[1],
                data.attribute(column2Index).name(), stats2[1]);
        System.out.printf("Mean:     %s = %.3f, %s = %.3f\n",
                data.attribute(column1Index).name(), stats1[2],
                data.attribute(column2Index).name(), stats2[2]);
        System.out.printf("Std Dev:  %s = %.3f, %s = %.3f\n",
                data.attribute(column1Index).name(), stats1[3],
                data.attribute(column2Index).name(), stats2[3]);
    }

    private static double[] calculateColumnStats(Instances data, int columnIndex) {
        if (data.attribute(columnIndex).isNumeric()) { // Check for numeric column
            double min = data.attributeStats(columnIndex).numericStats.min;
            double max = data.attributeStats(columnIndex).numericStats.max;
            double mean = data.attributeStats(columnIndex).numericStats.mean;
            double stdDev = data.attributeStats(columnIndex).numericStats.stdDev;
            return new double[] { min, max, mean, stdDev };
        } else {
            return null; // Indicate non-numeric column if needed
        }
    }

    private static void compareDatasets(Scanner scanner) throws Exception {
        System.out.println("\nEnter the number of datasets to compare:");
        int numDatasets = scanner.nextInt();
        scanner.nextLine(); // Consume newline character

        List<Instances> datasets = new ArrayList<>();

        for (int i = 0; i < numDatasets; i++) {
            System.out.println("Enter the path to dataset " + (i + 1) + " (CSV format):");
            String filePath = scanner.nextLine();

            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(filePath));
            Instances dataset = loader.getDataSet();
            datasets.add(dataset);
        }

        // Display column names for each dataset
        for (int i = 0; i < datasets.size(); i++) {
            System.out.println("\nColumn Names for Dataset " + (i + 1) + ":");
            for (int j = 0; j < datasets.get(i).numAttributes(); j++) {
                System.out.println("Index " + j + ": " + datasets.get(i).attribute(j).name());
            }
        }

        // Taking input from user to enter comparisons
        System.out.println("\n How many column comparisons would you like to do?");
        int numOfColumns = scanner.nextInt();
        scanner.nextLine();

        // Loop to get column pairings for comparison
        for (int i = 0; i < numOfColumns; i++) {
            System.out.println("\nComparison Pair " + (i + 1) + ":");
            List<Integer> datasetIndices = new ArrayList<>();
            List<Integer> columnIndices = new ArrayList<>();

            // Getting dataset and column indexes from user
            System.out.println("How many columns would you like to compare in this pair?");
            int numOfColumnsToCompare = scanner.nextInt();

            for (int j = 0; j < numOfColumnsToCompare; j++) {
                System.out.print("Enter the dataset index (1-" + numDatasets + ") for the column: ");
                int datasetIndex = scanner.nextInt() - 1; // Adjust for 0-based indexing
                datasetIndices.add(datasetIndex);

                System.out.print("Enter the column index in dataset " + (datasetIndex + 1) + ": ");
                int columnIndex = scanner.nextInt();
                columnIndices.add(columnIndex);
                scanner.nextLine();
            }

            // Input validation (similar to before, but now checking all columns in the
            // pairing)
            for (int j = 0; j < numOfColumnsToCompare; j++) {
                if (datasetIndices.get(j) < 0 || datasetIndices.get(j) >= numDatasets ||
                        columnIndices.get(j) < 0
                        || columnIndices.get(j) >= datasets.get(datasetIndices.get(j)).numAttributes()) {
                    System.out.println("Error: Invalid dataset or column index.");
                    return;
                }

                // Check if the column is numeric
                if (!datasets.get(datasetIndices.get(j)).attribute(columnIndices.get(j)).isNumeric()) {
                    System.out.println("Error: All columns must be numeric for comparison.");
                    return;
                }
            }

            // Perform comparison (now iterating through all selected columns)
            for (int j = 0; j < numOfColumnsToCompare; j++) {
                double[] stats = calculateColumnStats(datasets.get(datasetIndices.get(j)), columnIndices.get(j));

                System.out.println("\nColumn " + (j + 1) + ": "
                        + datasets.get(datasetIndices.get(j)).attribute(columnIndices.get(j)).name()
                        + " (Dataset " + (datasetIndices.get(j) + 1) + ")");
                System.out.println("Minimum: " + stats[0]);
                System.out.println("Maximum: " + stats[1]);
                System.out.println("Mean: " + stats[2]);
                System.out.println("Standard Deviation: " + stats[3]);
            }
        }
    }

    private static void predictWithJ48(Instances data, Scanner scanner, J48 j48Classifier) {
        try {
            // Load a new unlabelled dataset from user input
            System.out.print("Enter the path to the new unlabelled dataset (in CSV format): ");
            String datasetPath = scanner.next();

            // Load CSV file using CSVLoader
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(datasetPath));
            Instances newData = loader.getDataSet();

            // Set class index if needed
            if (newData.classIndex() == -1 && newData.numAttributes() > 0) {
                newData.setClassIndex(newData.numAttributes() - 1);
            }

            // Make predictions for the new dataset
            System.out.println("Predictions for the new unlabelled dataset:");
            for (int i = 0; i < newData.numInstances(); i++) {
                double predictedClass = j48Classifier.classifyInstance(newData.instance(i));
                System.out.println("Instance " + (i + 1) + ": Predicted class - "
                        + data.classAttribute().value((int) predictedClass));
            }

            // Calculate evaluation metrics
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(j48Classifier, newData);

            // Ask user if they want to see evaluation metrics
            System.out.print("Do you want to see evaluation metrics? (yes/no): ");
            String showEvaluation = scanner.next().toLowerCase();

            if (showEvaluation.equals("yes")) {
                // Show evaluation metrics
                System.out.println(eval.toSummaryString());
                System.out.println(eval.toMatrixString());
                System.out.println(eval.toClassDetailsString());
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
