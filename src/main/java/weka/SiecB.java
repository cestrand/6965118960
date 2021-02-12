/**
 * LogScore is
 * calculated by adding the minus logarithm of the probability assigned by the classifier to the correct class and gives
 * an idea of how well the classifier is estimating probabilities
 * (the smaller the score the better the result). -- https://www.aaai.org/Papers/FLAIRS/2003/Flairs03-066.pdf
 */

package weka;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.bayes.net.BIFReader;
import weka.classifiers.bayes.net.search.local.K2;
import weka.core.Instance;
import weka.core.Instances;

public class SiecB extends BIFReader {
	private SiecB() {

	}

	public static SiecB createK2(int maxNrOfParents) {
		SiecB r = new SiecB();
		((K2) r.getSearchAlgorithm()).setMaxNrOfParents(maxNrOfParents);
		return r;
	}

	public static SiecB createHillClimbLocal(int maxNrOfParents) {
		return createHillClimbLocal(maxNrOfParents, false);
	}

	public static SiecB createHillClimbLocal(int maxNrOfParents, boolean useArcReversal) {
		SiecB r = new SiecB();
		weka.classifiers.bayes.net.search.local.HillClimber hc = new weka.classifiers.bayes.net.search.local.HillClimber();
		hc.setMaxNrOfParents(maxNrOfParents);
		if(useArcReversal) {
			hc.setUseArcReversal(true);
		}
		r.setSearchAlgorithm(hc);
		return r;
	}


	public static SiecB createHillClimbGlobal(int maxNrOfParents) {
		return createHillClimbGlobal(maxNrOfParents, false);
	}

	public static SiecB createHillClimbGlobal(int maxNrOfParents, boolean useArcReversal) {
		SiecB r = new SiecB();
		weka.classifiers.bayes.net.search.global.HillClimber hc = new weka.classifiers.bayes.net.search.global.HillClimber();
		hc.setMaxNrOfParents(maxNrOfParents);
		if(useArcReversal) {
			hc.setUseArcReversal(true);
		}
		r.setSearchAlgorithm(hc);
		return r;
	}

	public static SiecB loadFromBIFAndNormalize(String filePath) throws Exception {
		SiecB r = new SiecB();
		r.processFile(filePath);
		r.normalizeDataSet(r.m_Instances);		//zamiast procesu uczenia ??
		return r;
	}

	/**
	 * Oblicza tabelę CPT (prawdopodobieństwa warunkowego).
	 * @param dane
	 * @throws Exception
	 */
	public void estimateCPTs(Instances dane) throws Exception {
		// estimateCPTs szacuje tabele prawdopodobieństwa warunkowego dla sieci Bayes
		// przy użyciu struktury sieci.
		// Conditional Probability Tables
		estimateCPTs(dane, false); // czyli - CPTs "od zera"
	}

	/**
	 * Oblicza tabelę CPT (prawdopodobieństwa warunkowego).
	 * @param dane
	 * @param add	Jeśli false to inicjalizuje CPT od nowa.
	 * @throws Exception
	 */
	public void estimateCPTs(Instances dane, boolean add) throws Exception {
		if (!add)
			this.initCPTs(); // tworzy i inicjuje NOWE tablice CPTs
		for (int j = 0; j < dane.numInstances(); j++) {
			this.updateClassifier(dane.get(j));
		}
	}


	/**
	 * Zwraca prawdopodobieństwo warunkowe danego atrybutu przy ustalonych innych wartościach atrybutu.
	 * @param attr
	 * @param vals
	 * @return
	 */
	public double condProb(int attr, double... vals) {
		double iCPT = 0; // indeks w tablicy prawdop. w-kowych dla H -dla atryb nr 2
		int nrOfParents = getParentSet(attr).getNrOfParents();
		for (int i = 0; i < nrOfParents; i++) {
			int nrR = getParentSet(attr).getParent(i);
			iCPT = iCPT * m_Instances.attribute(nrR).numValues() + vals[nrR];
		}
		return m_Distributions[attr][(int) iCPT].getProbability(vals[attr]);
	}

	/**
	 * Funkcja zwracająca prawdopodobieństwo warunkowe tzn. prawdopodobieństwo wystąpienia takiego rekordu w danych
	 * który miałby atrybuty vals ustawione na takie jak przekazane w argumencie.
	 * Jeżeli chcemy dopuścić wszystkie możliwe wartości danego atrybutu, nie ustawiać konkretną wartość to podajemy -1.
	 * @param vals Lista wartości atrybutów w kolejności takiej jak w danych.
	 *                Musi być długości dokładnie takiej ile jest atrybutów w danych.
	 * @return	Liczba z przedziału [0,1] określająca prawdopodobieństwo wystąpienia takiego rekordu
	 * na podstawie wytrenowanego modelu.
	 */
	public double prob(double... vals) {
		// rozklad brzegowy p (overcast, ?, high,?, ? = prob(1, -1, 0, -1, -1)
		// wartosci ujemne - > do przesumowania
		// czyli np. prop(overcast, hot, high, FALSE, yes) = prob(1,0,0,1,0)
		for (int i = 0; i < vals.length; i++) {
			if (vals[i] < 0) { // atrybut i-ty ma byc pominiety, tzn
				// dla kazdej wartosci i p(...,i,..) ma byc zsumowane
				double[] kopiaVals = Arrays.copyOf(vals, vals.length);
				double suma = 0;
				for (int j = 0; j < m_Instances.attribute(j).numValues(); j++) {
					kopiaVals[i] = j;
					suma = suma + prob(kopiaVals);
				}
				return suma;
			}
		}
		double il = 1;
		for (int i = 0; i < m_Instances.numAttributes(); i++)
			il = il * condProb(i, vals);
		return il;
	}

	/**
	 * Wykonuje sprawdzian krzyżowy polegający na wykonaniu klasyfikacji i-tego rekordu na sieci
	 * wytrenowanej bez i-tego rekordu.
	 * @param dane
	 * @return Tablica decyzji o długości takiej jak ilość wierszy w dane.
	 * @throws Exception
	 */
	public double[] xOneOut(Instances dane) throws Exception { // tablica decyzji
		// https://pl.wikipedia.org/wiki/Sprawdzian_krzy%C5%BCowy
		this.estimateCPTs(dane);
		int N = dane.numInstances();
		double[] dec = new double[N];
		for (int i = 0; i < N; i++) {
			Instance wiersz = dane.get(i);
			wiersz.setWeight(-wiersz.weight()); // usuwa wiersz z danych i z tablic CPTs
			this.updateClassifier(wiersz); // wylacza obsluge, odejmuje od calosci
			dec[i] = this.classifyInstance(wiersz); // nowa decyzja
			wiersz.setWeight(-wiersz.weight()); // teraz >0, powraca wiersz do danych i do tablic CPTs
			this.updateClassifier(wiersz); // wlacza obsluge, dodaje od calosci
		}
		return dec;
	}


	// x-walidacja "klasyczna", wersja domyslna
	public double[] xVal(Instances dane, int q) throws Exception {
		// W tej metodzie, oryginalna próba jest dzielona na K podzbiorów. Następnie
		// kolejno każdy z nich bierze się jako zbiór testowy, a pozostałe razem jako
		// zbiór uczący i wykonuje analizę. Analiza jest więc wykonywana K razy.
		// K rezultatów jest następnie uśrednianych (lub łączonych w inny sposób) w celu
		// uzyskania jednego wyniku.
		return xVal(dane, q, new Random());
	}

	private double[] restore(double[] dec, int[] perm) { // odtwarza w pierwotnej kolejnosci
		double[] rest = new double[dec.length];
		for (int i = 0; i < dec.length; i++)
			rest[perm[i]] = dec[i];
		return rest;
	}

	private int[] mix(Instances dX, Random los) { // miesza i zapamietuje kolejnosc
		int n = dX.numInstances();
		int[] perm = new int[n];
		for (int i = 0; i < n; i++) // oryginalna, poczatkowa kolejnosc wierszy
			perm[i] = i;
		for (int i = n - 1; i > 0; i--) { // ostatni zamieniamy z wylosowanym/ dla i od n-1 w dol
			int nr = perm[i]; // i-ty <-> losowy z zakresu 0,...,i
			int j = los.nextInt(i + 1);
			dX.swap(i, j); // mieszanie
			perm[i] = perm[j]; // stosowna zamiana numerow
			perm[j] = nr;
		}
		return perm;
	}

	/**
	 * Wykonuje sprawdzian krzyżowy. q-krotna walidacja. W tej metodzie, oryginalna próba jest dzielona na q podzbiorów.
	 * Następnie kolejno każdy z nich bierze się jako zbiór testowy, a pozostałe razem jako zbiór uczący
	 * i wykonuje analizę.
	 * @param dane
	 * @param q Ilość zbiorów na które ma zostać podzielony zbiór dane w tym sprawdzianie krzyżowym
 	 * @param los Stan generatora liczb pseudolosowych. W zasadzie tylko w celach testowych.
	 * @return Tablica decyzji o długości takiej jak ilość wierszy w dane.
	 * @throws Exception
	 */
	public double[] xVal(Instances dane, int q, Random los) throws Exception {
		// tablica decyzji w x-walidacji q-krotnej // q ziorow testowych
		// dla q=1. - test na danych uczacych
		int n = dane.numInstances();
		double[] dec = new double[n];
		Instances dX = new Instances(dane); // kopia - dla zachowania kolejnosci
//		dX.randomize(los); 
//		new int[n]; //tu: 0,1,2,3,... 
		int[] perm = mix(dX, los); // zamiast randomize, mieszanie
		for (int k = 0; k < q; k++) { // k-ta grupa; podzial na q grup;
			int start = (n * k) / q; // dla k-tej grupy; start = n*(k/q)
			int end = (n * (k + 1)) / q;
			// [0...)[start...end)[end...n), srodek - dla testow; reszta = zb. uczacy
			if (q > 1)
				initCPTs();
			for (int i = 0; i < start; i++) {
				updateClassifier(dX.get(i));
			}
			for (int i = end; i < n; i++) {
				updateClassifier(dX.get(i));
			}
			for (int i = start; i < end; i++) { // test dla kolejnej grupy rozmiaru n/q
				dec[i] = classifyInstance(dX.get(i)); // tu jest test
			}
		}
		this.estimateCPTs(dane); // na koniec budowa CPTs dla TYCH danych
		return restore(dec, perm); // tablica decyzji dla uporzadkowanych permutacji
	}

	/**
	 * Wykonuje xVal i potem zwraca macierz wizualizującą efekt tego sprawdzianu krzyżowego.
	 * @param dane
	 * @param q	Na ile podzbiorów wykonać podział w sprawdzianie krzyżowym. Gdy q = 0 to robi xOneOut
	 * @return Macierz wymiaru numClasses na numClasses
	 * @throws Exception
	 */
	public int[][] confMatrixOfXVal(Instances dane, int q) throws Exception {
		return confMatrixOfXVal(dane, q, new Random());
	}


	public int[][] confMatrixOfXVal(Instances dane, int q, Random los) throws Exception {
		double[] org = dane.attributeToDoubleArray(dane.classIndex());
		double[] dec;
		if (q == 1) { // na danych uczacych
			dec = xVal(dane, 1); // !?
		} else if (q == 0) { // OneOut
			dec = xOneOut(dane);
		} else {
			dec = xVal(dane, q, los); // dla q>=2: zgodna kolejnosc tablic dec i org
		}
		return Matrix.confMatrix(org, dec, m_Instances.numClasses());
	}

	/**
	 * Zapisuje model do pliku BIF.
	 * @param filePath Scieżka do pliku wynikowego. Powinna kończyć się na ".xml"
	 * @throws IOException
	 */
	public void saveToBIF(String filePath) throws IOException {
		FileWriter fw = null;
		if (!filePath.toLowerCase().endsWith(".xml"))
			filePath += ".xml";
		fw = new FileWriter(filePath);
		fw.write(this.toXMLBIF03());
		fw.close();
	}

	/**
	 * Wypisuje na konsolę informacje o klasyfikatorze.
	 */
	public void PrintScore() {
		System.out.println(this);
	}

	/**
	 * Wypisuje na konsolę dane.
	 */
	public void PrintData() {
		System.out.println(this.m_Instances);
	}

	public double[] predict(Instances dane) throws Exception {
		double[] predykcje = new double[dane.numInstances()];
		for(int i = 0; i < predykcje.length; i++) {
			predykcje[i] = classifyInstance(dane.get(i));
		}
		return predykcje;
	}

	public int[][] confMatrix(Instances dane) throws Exception {
		int[][] M = Matrix.confMatrix(dane.attributeToDoubleArray(dane.classIndex()), predict(dane), dane.numClasses());
		return M;
	}
}
