package dataset.diabetes;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.*;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize; //dyskretyzacja nie beirze pod uwage wskazanego atrybutu decyzyjnego

public class Diabetes {

	// Blok 1
	// Jak utworzyc klasyfikator (lub klasyfikatory polaczone w calosc)
	// Typ klasyfikatore: SiecB i parametry
	// (wyj�tkowo - TU:) Powstaly 2 zbiory danych : dobre i inne, na ktorych (na kazdym) siecB o danej
	// strukturze dziala dobrze
	// a) TE klasyfikatory utworzyc i ZAPISAC
	// b) stworzyc odredny klasyfikator dla klas: "dobre" i "inne"

	// Co (i jak) z wynikow mozna zapisac - do szybkiego odtworzenia
	// np. strukture sieci B. jako pliki BIFF i...?

	// Blok 2
	// Jak korzystac z utworzonego systemu
	// odtworzyc siec(-i) -> jako gotowy klasyfikator
	// a) Znalezc dobry klasyfikator dla nowego zbioru DiaDI.arff (TU s� dwie "odr�bne grupy danych")
	// b) Z 2 plikow BIFF odtworzyc dwa klasyfikatory typu SiecB
	// Zapami�ta� filtr dyskretyzuj�cy - raczej jako kod w javie

	// d)jak klasyfikowac nowe dane?
	// liczby przepisac na instancje// pozniej ewentualny test

	public static void main(String[] args) throws Exception {
		przygotowanie();
		rekreacja();
	}

	private static void rekreacja() throws Exception{
		// II Odtworzy� to, co zapisane (i to co konieczne)
		Instances dane = DataSource.read("zasoby/weka-data/diabetes.arff");
		Instances daneOrg = new Instances(dane);

		dane.setClassIndex(dane.numAttributes()-1);

		// Filtry - dwie dyskretyzacje
		dane = Data.discretizeAttributesUnsupervised(dane, "3-4", 4, true);

		daneOrg = Data.discretizeAttributesSupervised(dane, "3-4");

		SiecB diaDobre = SiecB.loadFromBIFAndNormalize("zasoby/DiaDobre.xml");
		SiecB diaInne = SiecB.loadFromBIFAndNormalize("zasoby/DiaInne.xml");

		Instance wiersz = dane.instance(2);
		double predykcja = diaDobre.classifyInstance(wiersz);
		predykcja = diaInne.classifyInstance(wiersz);
	}

	private static void przygotowanie() throws Exception, IOException {
		Instances dane = DataSource.read("zasoby/weka-data/diabetes.arff");
		Instances daneOrg = dane; // TO SAMO a nie kopia - dane beda dyskretyzowane
		// ewentualna kopia: = new Instances(dane);

		daneOrg.setClassIndex(dane.numAttributes() - 1);
		Attribute klasa = dane.classAttribute();

		dane = Data.discretizeAttributesUnsupervised(dane,"3-4", 4, true);

		// klasyfikacja
		SiecB bn = SiecB.createHillClimbGlobal(2);
		bn.buildClassifier(dane);
		bn.PrintScore();
		bn.PrintData();


		// Sprawdźmy jak dobra jest klasyfikacja
		// Wielokrotnie wykonajmy x-walidacje dzielącją zbiór na dwa
		int numOfXVals = 500;
		MultipleXVal xval = new MultipleXVal(numOfXVals, 2, dane, bn);
		xval.Run();

		// Dokojanmy podziału na outliery i zwykłe dane.
		ArrayList<Integer> skrajneIndeksy = xval.InstancesWithErrorsOver((int) 0.6*dane.numInstances());
		System.out.println("Skrajnych: " + skrajneIndeksy.size());

		Podzial podzial = Data.Split(skrajneIndeksy, dane);
		Instances inne = podzial.l;
		Instances dobre = podzial.r;

		dobre.setRelationName("Diabetes_dobre");
		DataSink.write("zasoby/DiaDobre.arff", dobre);
		inne.setRelationName("Diabetes_inne");
		DataSink.write("zasoby/DiaInne.arff", inne);

		// xWalidacja OneOut bo q = 0
		Matrix.show(bn.confMatrixOfXVal(dobre, 0));
		bn.saveToBIF("zasoby/DiaDobre.xml");

		Matrix.show(bn.confMatrixOfXVal(inne, 0));
		bn.saveToBIF("zasoby/DiaInne.xml");


		dobre.setClassIndex(-1);
		inne.setClassIndex(-1);
		dobre.deleteAttributeAt(dane.classIndex());
		inne.deleteAttributeAt(dane.classIndex());

		Instances daneDI = Data.DodajKolumneWzgledemPodzialu(podzial, "DobreInne", "inny", "dobry");
		daneDI.setClassIndex(daneDI.numAttributes() - 1);


		daneDI.setRelationName("Diabetes-DobreInne");
		DataSink.write("zasoby/DiaDI.arff", daneDI);
	}
}