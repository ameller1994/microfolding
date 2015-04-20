import java.io.*;
import java.util.*;
import com.google.common.collect.*;

/**
 * This class reads parameters from the OPLS force field file and stores them in various maps.
 */
public class OPLSforcefield implements Singleton
{
    /** Map from OPLS atom types to atom classes. */
    public static final Map<Integer,Integer> CLASS_MAP;

    /** Map from atom classes to VDW distances (sigma) in A. */
    public static final Map<Integer,Double> VDW_DISTANCE_MAP;

    /** Map from atom classes to VDW depths (epsilon) in kcal/mol. */
    public static final Map<Integer,Double> VDW_DEPTH_MAP;

    /** Map from atom types to partial charges. */
    public static final Map<Integer,Double> CHARGE_MAP;

    /** Map from atom class to torsional parameters/ */
    public static final Map<List<Integer>, TorsionalParameter> TORSIONAL_MAP;

    private OPLSforcefield()
    {
        throw new IllegalArgumentException("not instantiable");
    }

    static
        {
            // make temporary fields
            Map<Integer,Integer> tempClassMap = new HashMap<>();
            Map<Integer,Double> tempDistanceMap = new HashMap<>();
            Map<Integer,Double> tempDepthMap = new HashMap<>();
            Map<Integer,Double> tempChargeMap = new HashMap<>();
            Map<List<Integer>, TorsionalParameter> tempTorsionalMap = new HashMap<>();

            // read forcefield file
            OutputFileFormat forcefieldFile = new OutputFileFormat(Settings.OPLS_FORCEFIELD_FILENAME) {};

            for (List<String> fields : forcefieldFile.fileContents)
                {
                    if ( fields.size() == 0 )
                        continue;
                    if      ( fields.get(0).equals("atom")  && fields.size() >= 3 )
                        {
                            Integer typeNumber = Integer.valueOf(fields.get(1));
                            Integer classNumber = Integer.valueOf(fields.get(2));
                            if ( tempClassMap.containsKey(typeNumber) )
                                throw new IllegalArgumentException("duplicate type/class line\n" + fields.toString());
                            tempClassMap.put(typeNumber, classNumber);
                        }
                    else if ( fields.get(0).equals("vdw")   && fields.size() >= 4 )
                        {
                            Integer classNumber = Integer.valueOf(fields.get(1));
                            Double distance     = Double.valueOf(fields.get(2));
                            Double depth        = Double.valueOf(fields.get(3));
                            if ( tempDistanceMap.containsKey(classNumber) || tempDepthMap.containsKey(classNumber) )
                                throw new IllegalArgumentException("duplicate vdw line\n" + fields.toString());
                            tempDistanceMap.put(classNumber, distance);
                            tempDepthMap.put(classNumber, depth);
                        }
                    else if ( fields.get(0).equals("charge") && fields.size() >= 3 )
                        {
                            Integer type   = Integer.valueOf(fields.get(1));
                            Double  charge = Double.valueOf(fields.get(2));
                            if ( tempChargeMap.containsKey(type) )
                                throw new IllegalArgumentException("duplicate charge line\n" + fields.toString());
                            tempChargeMap.put(type,charge);
                        }
                    /* else if ( fields.get(0).equals("bond") && fields.size() >= 3 )
                        {
                            Integer classNumber1 = Integer.valueOf(fields.get(1));
                            Integer classNumber2 = Integer.valueOf(fields.get(2));

                            Double k = Double.valueOf(fields.get(3));
                            Double r0 = Double.valueOf(fields.get(4));
                        }
                    else if ( fields.get(0).equals("angle") && fields.size() >= 3 )
                        {
                            Integer classNumber1 = Integer.valueOf(fields.get(1));
                            Integer classNumber2 = Integer.valueOf(fields.get(2));
                            Integer classNumber3 = Integer.valueOf(fields.get(3));

                            Double k = Double.valueOf(fields.get(4));
                            Double theta0 = Double.valueOf(fields.get(5));
                        }
                    */
                    else if ( fields.get(0).equals("torsion")) // || fields.get(0).equals("imptors")) && fields.size() >= 3 )
                        {
                            List<Integer> atomClasses = new LinkedList<>();
                            atomClasses.add(Integer.valueOf(fields.get(1)));
                            atomClasses.add(Integer.valueOf(fields.get(2)));
                            atomClasses.add(Integer.valueOf(fields.get(3)));
                            atomClasses.add(Integer.valueOf(fields.get(4)));

                            // differing lengths are possible for the torsion parameter data 
                            double[] amplitudes = new double[3];
                            for (int i = 5; i < fields.size(); i = i+3) 
                            {
                                //ignore comments
                                if (fields.get(i).equals("#"))
                                    break;
                                int periodicity = Integer.valueOf(fields.get(i+2));
                                amplitudes[periodicity-1] = Double.valueOf(fields.get(i));

                            }
                            
                            // if there is no data, assign zero energy to this torsion

                            TorsionalParameter torsionalParameter = new TorsionalParameter(amplitudes);
                            tempTorsionalMap.put(atomClasses, torsionalParameter);

                            //note there is one duplicate value in the OPLS parameter file
                        }

                }

            // override charges for CA and N of backbones
            // all are set to the alanine parameters
            // vdw types don't need changing because they're all the same atom class
            //tempChargeMap.put(93, 0.14);   // proline HA
            //tempChargeMap.put(86,-0.14);   // proline N
            //tempChargeMap.put(73, 0.14);   // glycine HA

            // set permanent fields
            CLASS_MAP = ImmutableMap.copyOf(tempClassMap);
            VDW_DISTANCE_MAP = ImmutableMap.copyOf(tempDistanceMap);
            VDW_DEPTH_MAP = ImmutableMap.copyOf(tempDepthMap);
            CHARGE_MAP = ImmutableMap.copyOf(tempChargeMap);
            TORSIONAL_MAP = ImmutableMap.copyOf(tempTorsionalMap);
        }

    /** A class representing the torsional parameters of a dihedral angle. 
     * Dihedral energies are calculated as E = sigma( Vi/2*(1+cos*Per_i(phi - Phase_i)) ) where V is the amplitude, Per is the periodicity.
     * The OPLS forcefield only contains a Fourier expansion up to 3 terms */
    public static class TorsionalParameter implements Immutable
    {
        /** The amplitudes for a Fourier term */
        public final double[] amplitudes;

        /** Constructor that assumes user passes in parallel lists of torsional parameter terms. 
         * Note that the phase for all odd terms is assumed to be 0.0 degrees and the phase for even terms is 180.0 
         */ 
        public TorsionalParameter(double[] amplitudes)
        {
            if (amplitudes.length != 3)
                throw new IllegalArgumentException("The number of torsional terms is not 3");
            this.amplitudes = amplitudes;
        }

        @Override
        public String toString()
        {
            String returnString = "Torsional parameter: \n";
            for (int i = 0; i < amplitudes.length; i++)
            {
                double phase = 0.0;
                // Even terms (i is odd) have phase of 180 degrees in OPLS data file
                if (i % 2 != 0)
                    phase = 180.0;
                returnString = returnString + "P: " + (i+1) + " A: " + amplitudes[i] + " Ph: " + phase + "\n";  
            }
            return returnString;
        }
    }

    /** Forces the database to load. */
    public static void load()
    {
        System.out.printf("%d OPLS atom types have been loaded.\n", CLASS_MAP.size());
        System.out.printf("%d OPLS vdw classes have been loaded.\n", VDW_DISTANCE_MAP.size());
        System.out.printf("%d OPLS charge parameters have been loaded.\n", CHARGE_MAP.size());
        System.out.printf("%d OPLS torsional parameters have been loaded. \n", TORSIONAL_MAP.size());
    }

    /** For testing. */
    public static void main(String[] args)
    {
        load();
        // Test for specific values
        int class1 = 1;
        int class2 = 1;
        int class3 = 1;
        int class4 = 2;
        List<Integer> atomClasses =  new LinkedList<>();
        atomClasses.add(class1);
        atomClasses.add(class2);
        atomClasses.add(class3);
        atomClasses.add(class4);

        TorsionalParameter torsionalParameter = FixedSequenceOPLScalculator.getTorsionalParameter(atomClasses);
        System.out.println(torsionalParameter);
    }
} 
