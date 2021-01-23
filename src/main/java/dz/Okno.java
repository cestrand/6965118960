package dz;

import weka.core.Instances;

import javax.swing.*;
import java.awt.*;

public class Okno extends JFrame {

    public Okno() {
        String[] columnNames = {"First Name",
                "Last Name",
                "Sport",
                "# of Years",
                "Vegetarian"};

        Object[][] data = {
                {"Kathy", "Smith",
                        "Snowboarding", 5, false},
                {"John", "Doe",
                        "Rowing", 3, true},
                {"Sue", "Black",
                        "Knitting", 2, false},
                {"Jane", "White",
                        "Speed reading", 20, true},
                {"Joe", "Brown",
                        "Pool", 10, false}
        };

        final JTable table = new JTable(data, columnNames);
        table.setPreferredScrollableViewportSize(new Dimension(500, 70));
        table.setFillsViewportHeight(true);
        JScrollPane scrollPane = new JScrollPane(table);

        //Add the scroll pane to this panel.
        add(scrollPane);

        pack();

        setVisible(true);

    }

    public static void main(String[] args) {
        //Schedule a job for the event-dispatching thread:
        //creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                Okno o = new Okno();
            }
        });
    }
}
