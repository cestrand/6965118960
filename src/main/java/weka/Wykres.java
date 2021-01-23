package weka;/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYDotRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.DefaultXYDataset;

/**
 *
 * @author Grzegorz
 */
public class Wykres {
   ChartFrame cf;
   DefaultXYDataset ds = new DefaultXYDataset();
   
   public Wykres(String tytul) {
	   this(tytul,false); 		//tylko punkty
   }
   public Wykres(String tytul,boolean lines) {
      NumberAxis xAxis = new NumberAxis("os X");
 //     xAxis.setAutoRangeIncludesZero(false);
      NumberAxis yAxis = new NumberAxis("os Y");
      XYPlot plot;
      if (lines) {
    	  XYItemRenderer rend = new XYLineAndShapeRenderer(true, false);
    	  plot = new XYPlot(ds, xAxis, yAxis, rend);
      } else {
	      XYDotRenderer rend = new XYDotRenderer();
	      rend.setDotHeight(2);
	      rend.setDotWidth(2);
	      plot = new XYPlot(ds, xAxis, yAxis, rend);
      }
      
      plot.setOrientation(PlotOrientation.VERTICAL);
      JFreeChart fc = new JFreeChart(plot);
        cf = new ChartFrame(tytul, fc);
   }

   public void pokaz(String opis, double[][] dane) {
        ds.addSeries(opis, dane);
        cf.pack();
        cf.setSize(600, 400);
        cf.setVisible(true);
   }
}