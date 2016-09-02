package com.enderunlabs.app;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Ml;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.awt.*;
import java.awt.Point;
import java.io.File;
import java.io.IOException;
import java.util.*;



/**
 * Created by Lenovo on 23.08.2016.
 */
public class OCR {

    public final char[] strCharacters = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L',
            'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'};
    public final int numCharacters = 30;
    public int charSize;
    public boolean DEBUG;
    public boolean saveSegments;
    public String filename;
    private int K;
    private boolean trained;
    private ANN_MLP ann;

    // Define constants
    public static final int HORIZONTAL = 1;
    public static final int VERTICAL = 0;


    public OCR(String trainFile) throws ParserConfigurationException, IOException, SAXException {
        trained = false;
        charSize = 20;
        DEBUG = false;
        saveSegments = false;

        File f = new File("src/main/resources/xml/OCR.xml");

        DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        Document doc = dBuilder.parse(f);

        doc.getDocumentElement().normalize();
        NodeList nList = doc.getElementsByTagName("TrainingDataF5");
        NodeList nListTwo = doc.getElementsByTagName("classes");

        Mat trainingData = null;
        Mat classes = null;

        fillMat(trainingData, nList);
        fillMat(classes, nListTwo);

        train(trainingData,classes,10);

    }

    public OCR() {
        trained = false;
        charSize = 20;
        saveSegments = false;
        DEBUG = false;
    }

    public Mat backProjection(Mat image_gray){
        java.util.List<Mat> matList = new LinkedList<Mat>();
        matList.add(image_gray);
        Mat histogram = new Mat();
        MatOfFloat ranges=new MatOfFloat(0,256);
        Imgproc.calcHist(matList, new MatOfInt(0), new Mat(), histogram , new MatOfInt(25), ranges);
        return histogram;
    }

    public Mat preprocessChar(Mat in){
        int h = in.rows();
        int w = in.cols();
        Mat transformMat = Mat.eye(2,3,CvType.CV_32F);
        int m = Math.max(w,h);
        transformMat.put(0,2,m/2-w/2);
        transformMat.put(1,2,m/2-h/2);

        Mat warpImage = new Mat(m,m,in.type());
        Imgproc.warpAffine(in,warpImage, transformMat, warpImage.size(), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0));
        Mat out = new Mat();
        Imgproc.resize(warpImage,out,new Size(charSize,charSize));

        return out;
    }

    public boolean verifySizes(Mat r){

        float aspect = 45.0f/77.0f;
        float charAspect = (float)r.cols()/(float)r.rows();
        float error = 0.35f;
        float minHeight = 15;
        float maxHeight = 28;
        float minAspect=0.2f;
        float maxAspect=aspect+aspect*error;
        float area=Core.countNonZero(r);
        float bbArea=r.cols()*r.rows();
        float percPixels=area/bbArea;
        if (DEBUG)
            System.out.println("Aspect: " + aspect + " [" + minAspect + "," + maxAspect + "] " + "Area " + percPixels +
                    " Char Aspect " + charAspect + " Height Char " + r.rows() + "\n");
        if(percPixels < 0.8 && charAspect > minAspect && charAspect < maxAspect && r.rows() >= minHeight && r.rows() < maxHeight)
            return true;
        else
            return false;
    }

    public int classify(Mat f){
        int result = -1;

        Mat output = new Mat(1,numCharacters,CvType.CV_32FC1);
        ann.predict(f,output,result);

        Core.minMaxLoc(output);

        return result;
    }

    public void train(Mat trainData, Mat classes, int nLayers) {

        Mat layers = new Mat(1, 3, CvType.CV_32SC1);
        layers.put(0, 0, trainData.cols());
        layers.put(0, 1, nLayers);
        layers.put(0, 2, numCharacters);

        ann.create();
        ann.setLayerSizes(layers);
        ann.setActivationFunction(ANN_MLP.SIGMOID_SYM);
        // alpha, beta!

        Mat trainClasses = new Mat();
        trainClasses.create(trainData.rows(), numCharacters, CvType.CV_32FC1);

        for (int i = 0; i < trainClasses.rows(); i++) {
            for (int k = 0; k < trainClasses.cols(); k++) {
                int [] ClassesNumber =new int[1];

                classes.get(i, k);

                // ???
                if(k ==  ClassesNumber[0])
                    trainClasses.put(i, k);
                else
                    trainClasses.put(i, k);
            }
        }
        Mat weights = new Mat(1,trainData.rows(),CvType.CV_32FC1, Scalar.all(1));

        ann.train(trainData, Ml.ROW_SAMPLE ,weights);
        trained = true;
    }

        public static Mat fillMat (Mat data, NodeList nList){

            for (int i = 0; i < nList.getLength(); i++) {
                Node node = nList.item(i);

                if (node.getNodeType() == Node.ELEMENT_NODE) {
                    Element element = (Element) node;

                    String type_id = element.getAttribute("type_id");
                    if ("opencv-matrix".equals(type_id) == false) {
                        System.out.println("Fault type_id ");
                    }

                    String rowsStr = element.getElementsByTagName("rows").item(0).getTextContent();
                    String colsStr = element.getElementsByTagName("cols").item(0).getTextContent();
                    String dtStr = element.getElementsByTagName("dt").item(0).getTextContent();
                    String dataStr = element.getElementsByTagName("data").item(0).getTextContent();

                    int rows = Integer.parseInt(rowsStr);
                    int cols = Integer.parseInt(colsStr);
                    int type = CvType.CV_8U;

                    Scanner s = new Scanner(dataStr);
                    s.useLocale(Locale.US);

                    if ("f".equals(dtStr)) {
                        type = CvType.CV_32F;
                        data = new Mat(rows, cols, type);
                        float fs[] = new float[1];
                        for (int r = 0; r < rows; r++) {
                            for (int c = 0; c < cols; c++) {
                                if (s.hasNextFloat()) {
                                    fs[0] = s.nextFloat();
                                } else {
                                    fs[0] = 0;
                                    System.err.println("Unmatched number of float value at rows=" + r + " cols=" + c);
                                }
                                data.put(r, c, fs);
                            }
                        }
                    } else if ("i".equals(dtStr)) {
                        type = CvType.CV_32S;
                        data = new Mat(rows, cols, type);
                        int is[] = new int[1];
                        for (int r = 0; r < rows; r++) {
                            for (int c = 0; c < cols; c++) {
                                if (s.hasNextInt()) {
                                    is[0] = s.nextInt();
                                } else {
                                    is[0] = 0;
                                    System.err.println("Unmatched number of int value at rows=" + r + " cols=" + c);
                                }
                                data.put(r, c, is);
                            }
                        }
                    } else if ("s".equals(dtStr)) {
                        type = CvType.CV_16S;
                        data = new Mat(rows, cols, type);
                        short ss[] = new short[1];
                        for (int r = 0; r < rows; r++) {
                            for (int c = 0; c < cols; c++) {
                                if (s.hasNextShort()) {
                                    ss[0] = s.nextShort();
                                } else {
                                    ss[0] = 0;
                                    System.err.println("Unmatched number of int value at rows=" + r + " cols=" + c);
                                }
                                data.put(r, c, ss);
                            }
                        }
                    } else if ("b".equals(dtStr)) {
                        data = new Mat(rows, cols, type);
                        byte bs[] = new byte[1];
                        for (int r = 0; r < rows; r++) {
                            for (int c = 0; c < cols; c++) {
                                if (s.hasNextByte()) {
                                    bs[0] = s.nextByte();
                                } else {
                                    bs[0] = 0;
                                    System.err.println("Unmatched number of byte value at rows=" + r + " cols=" + c);
                                }
                                data.put(r, c, bs);
                            }
                        }
                    }
                }
            }
            return data;
        }
    }