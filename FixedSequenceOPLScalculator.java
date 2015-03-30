import java.io.*;
import java.util.*;
import com.google.common.collect.*;
import org.apache.commons.math3.geometry.euclidean.threed.*;
import org.jgrapht.*;
import org.jgrapht.graph.*;
import org.jgrapht.alg.*;

/**
 * This class enables quick calculation of a conformer's OPLS energy given a mutation and the previous state of the peptide.
 * This class takes advantage of the insight that mutations in dihedral angles do not affect angles and bond distances. 
 * A general formula that encapsulates this approach is E_conformer = E_previous + delta(E). 
 * This class is designed for use in fixed sequence Monte Carlo simulations with mutations involving dihedral rotations. 
 */
public class FixedSequenceOPLScalculator
{
    /**
     * the coulombic constant for calculating electrostatic interactions, which is calculated as
     * e*e/(4*pi*epsilon_0) in A and kcal/mol
     */
    //public static final double COULOMB_CONSTANT = 332.3814783;
    public static final double COULOMB_CONSTANT = 332.06;

    /** distances that are less than this value will be set to this value to avoid blowups */
    public static final double MIN_DISTANCE = 1.0;

    /** The current state of the peptide that corresponds to this calculator. 
     * Calculations of the energy change associated with a mutation will assume this is the previous state 
     */
    public Peptide currentConformation;

    /** The current conformation's non-bonded energy (electrostatics + sterics) which is stored to find the energy change for a mutation */
    public double currentNonBondedEnergy;

    /** The previous conformation which allows for reversing mutations if they are rejected by the Monte Carlo process */
    public Peptide previousConformation;
 
    /** The previous conformation's non-bonded energy (electrostatics + sterics) which is stored to find the energy change for a mutation.
     * It allows for reversal of mutations */
    public double previousNonBondedEnergy;

    /** Creates an OPLScalculator for a Monte Carlo simulation
     * @param startingPeptide the initial peptide which will be minimized with a Monte Carlo process */
    public FixedSequenceOPLScalculator(Peptide startingPeptide)
    {
        // Call OPLS calculation in Tinker
        TinkerJob initialEnergyCalculation = new TinkerJob(startingPeptide, Forcefield.OPLS, 1000, false, true, false, false, false);
        TinkerJob.TinkerResult tinkerResult = initialEnergyCalculation.call();
        this.currentConformation = tinkerResult.minimizedPeptide; 

        throw new IllegalArgumentException("not instantiable");
    }

    /** Changes the energy of the peptide for a dihedral angle change. 
     * This method queries the OPLS forcefield for the dihedral's corresponding OPLS parameters. 
     * It finds the energy change of a dihedral change and also the resulting change in steric and electrostatic energies 
     * @param protoTorsion the torsion that is being modified. The method assumes this ProtoTorsion is part of the current conformation. 
     * @param newValue the value that the mutated torsion will take on
     * @param newConformation the mutated peptide
     * @return the energy of the new confirmation 
     */
    public double modifyDihedral(ProtoTorsion protoTorsion, Double newValue, Peptide newConformation)
    {
        // Query forcefield
        // Atom numbers -> Atom classes -> Torsional Parameters

        // Get energy change assocaited with dihedral

        // Get energy change in sterics and electrostatics 
        // Only calculate change between the two ends of the proto torsion 


        return 0.0;
    }

    /** Returns the change in energy between the current dihedral value and the new value
     * This method queries the OPLS forcefield and finds the energy for the current conformation's dihedral and the new value's energy.
     * This method uses the formula E = sigma( Vi/2*(1+cos(Per_i*(phi - Phase_i))) ) where V is the amplitude, Per is the periodicity.
     * @param protoTorsion the proto torsion that is being mutated
     * @param newValue the new value for the proto tosion
     * @retun the energy change in the dihedral between the previous value and the updated value
     */
    private double getEnergyChangeInDihedral(ProtoTorsion protoTorsion, double newValue)
    {
        // Get torsional parameters for the proto torsion
        List<Integer> atomClasses = new LinkedList<>();
        atomClasses.add(OPLSforcefield.CLASS_MAP.get(protoTorsion.atom1.type2));
        atomClasses.add(OPLSforcefield.CLASS_MAP.get(protoTorsion.atom2.type2));
        atomClasses.add(OPLSforcefield.CLASS_MAP.get(protoTorsion.atom3.type2));
        atomClasses.add(OPLSforcefield.CLASS_MAP.get(protoTorsion.atom4.type2));
        OPLSforcefield.TorsionalParameter torsionalParameter = OPLSforcefield.TORSIONAL_MAP.get(atomClasses);

        // Perform calculation of energy change using the formula: E = sigma( Vi/2*(1+cos(Per_i*(phi - Phase_i)) )) where V is the amplitude, Per is the periodicity.
        double oldValue = protoTorsion.getDihedralAngle();
        double energyChange = 0.0;
        for (int i = 0; i < torsionalParameter.periodicity.size(); i++)
        {
            double E_i_old = torsionalParameter.amplitudes.get(i) / 2 * (1 + Math.cos(torsionalParameter.periodicity.get(i) * (oldValue - torsionalParameter.phase.get(i))));  
            double E_i_new = torsionalParameter.amplitudes.get(i) / 2 * (1 + Math.cos(torsionalParameter.periodicity.get(i) * (newValue - torsionalParameter.phase.get(i))));  
            energyChange = energyChange + (E_i_new - E_i_old);
        }

        return energyChange;

    }

    /** Returns the energy of a new conformation following a set of dihedral changes and rotamer packing. 
     * This method is meant to calculate the energy change for a fragment insertion or a fragment generation muation
     * It finds the energy change in each of the dihedral angles and the steric and electrostatic energy change
     * @param protoTosions the dihedral angles that are being modified within the current conformation
     * @param newValues the new values for the dihedrals. This list should be parallel to the list of dihedrals.
     * @param newConformation the conformation after rotamer packing which will be used to calculate new nonbonded energies
     */
    private double makeMutation(List<ProtoTorsion> protoTorsions, List<Double> newValues, Peptide newConformation)
    {
        // find change in dihedral energies

        // calculate steric and electrostatic energy change for entire molecule

    }

    /**
     * Gets the OPLS partial charge.
     * @param type the OPLS atom type
     * @return the partial charge in fractions of an electron charge
     */
    public static Double getCharge(int type)
    {
        Double charge = OPLSforcefield.CHARGE_MAP.get(type);
        if ( charge == null )
            throw new NullPointerException("charge not found for type " + type);
        return charge;
    }

    /**
     * Gets the VDW distance.
     * @param classNumber the OPLS atom class
     * @return the VDW potential distance in angstroms
     */
    public static Double getVDWDistance(Integer classNumber)
    {
        Double distance = OPLSforcefield.VDW_DISTANCE_MAP.get(classNumber);
        if ( distance == null )
            throw new NullPointerException("vdw not found for class " + classNumber);
        return distance;
    }

    /**
     * Gets the VDW distance.
     * @param classNumber the OPLS atom class
     * @return the VDW well depth in kcal/mol
     */
    public static Double getVDWDepth(Integer classNumber)
    {
        Double depth = OPLSforcefield.VDW_DEPTH_MAP.get(classNumber);
        if ( depth == null )
            throw new NullPointerException("vdw not found for class " + classNumber);
        return depth;
    }

    /**
     * Get the OPLS atom class.
     * @param type the OPLS atom type
     * @return the OPLS atom class
     */
    public static Integer getOPLSClass(int type)
    {
        Integer classNumber = OPLSforcefield.CLASS_MAP.get(type);
        if ( classNumber == null )
            throw new NullPointerException("class not found for type " + type);
        return classNumber;
    }

}
