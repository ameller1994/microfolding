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
 * The OPLS force field is developed in <a href="http://pubs.acs.org/doi/abs/10.1021/ja9621760">Jorgensen et al., 1996</a>. */
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
        this.currentNonBondedEnergy = getNonBondedEnergy(tinkerResult.minimizedPeptide);
        this.previousConformation = null;
        this.previousNonBondedEnergy = 0.0;
    }

    /** Calculates the energy of the peptide for a dihedral angle change. 
     * This method queries the OPLS forcefield for the dihedral's corresponding OPLS parameters. 
     * It finds the energy change of a dihedral change and also the resulting change in steric and electrostatic energies 
     * @param protoTorsion the torsion that is being modified. The method assumes this ProtoTorsion is part of the current conformation. 
     * @param newValue the value that the mutated torsion will take on
     * @param newConformation the mutated peptide
     * @return the energy of the new confirmation 
     */
    public double modifyDihedral(ProtoTorsion protoTorsion, Double newValue, Peptide newConformation)
    {
        // Get energy change assocaited with dihedral change
        double energyChange = getEnergyChangeInDihedral(protoTorsion, newValue);

        // Get energy change in sterics and electrostatics
        // Only calculate change between the two ends of the proto torsion 
        // First find current energy between two segments of dihedral
        Set<Atom> currentSet1 = currentConformation.getHalfGraph(protoTorsion.atom2, protoTorsion.atom1);
        Set<Atom> currentSet2 = currentConformation.getHalfGraph(protoTorsion.atom3, protoTorsion.atom4);
        double currentNonBondedEnergyBetweenSegments = getNonBondedEnergy(currentSet1, currentSet2, protoTorsion.atom1, protoTorsion.atom4);

        // Second find the energy between the two segments after the change in dihedral angle (mutation)
        int index1 = currentConformation.contents.indexOf(protoTorsion.atom1);
        int index2 = currentConformation.contents.indexOf(protoTorsion.atom2);
        Atom atom1 = newConformation.contents.get(index1);
        Atom atom2 = newConformation.contents.get(index2);
        Set<Atom> newSet1 = newConformation.getHalfGraph(atom2, atom1);

        int index3 = currentConformation.contents.indexOf(protoTorsion.atom3);
        int index4 = currentConformation.contents.indexOf(protoTorsion.atom4);
        Atom atom3 = newConformation.contents.get(index3);
        Atom atom4 = newConformation.contents.get(index4);
        Set<Atom> newSet2 = newConformation.getHalfGraph(atom3, atom4);

        // Find the non-bonded energy between the two sets of atoms after the mutation
        double newNonBondedEnergyBetweenSegments = getNonBondedEnergy(newSet1, newSet2, atom1, atom4);

        double energyChangeBetweenSegments = newNonBondedEnergyBetweenSegments - currentNonBondedEnergyBetweenSegments;
        energyChange += energyChange + energyChangeBetweenSegments;

        // Find change in solvation energy
        // Is there way to make this more efficient??? 
        
        /* double solvationEnergy = 0.0;
        List<Double> SASAlist = null;
        try { SASAlist = new DCLMAreaCalculator(0.0).calculateSASA(newPeptide); }
        catch (Exception e) { e.printStackTrace(); SASAlist = ShrakeRupleyCalculator.INSTANCE.calculateSASA(newPeptide); }
        for (int i=0; i < SASAlist.size(); i++)
        {
            double surfaceArea = SASAlist.get(i);
            double surfaceTension = peptide.contents.get(i).surfaceTension;
            double energy = surfaceArea * surfaceTension;
            //System.out.printf("%3d  %8.2f  %8.2f\n", i+1, surfaceArea, energy);
            solvationEnergy += energy;
        }*/
        double newSolvationEnergy = 0.0; //solvationEnergy;

        // Find new energies
        double newPotentialEnergy = currentConformation.energyBreakdown.potentialEnergy + energyChange;
        double totalEnergy = newSolvationEnergy + newPotentialEnergy;

        newConformation = newConformation.setEnergyBreakdown(new EnergyBreakdown(null, totalEnergy, newSolvationEnergy, newPotentialEnergy, null, Forcefield.OPLS));  
        
        // update conformations 
        previousConformation = currentConformation;
        previousNonBondedEnergy = currentNonBondedEnergy;
        currentConformation = newConformation;
        currentNonBondedEnergy = currentNonBondedEnergy + energyChangeBetweenSegments; 
        
        // NOTE THIS IS FOR TESTING
        return newPotentialEnergy;
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
        atomClasses.add(getOPLSClass(protoTorsion.atom1.type2));
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
        double energyChange = 0.0;
        for (int i = 0; i < protoTorsions.size(); i++)
            energyChange += getEnergyChangeInDihedral(protoTorsions.get(i), newValues.get(i));
        

        // calculate steric and electrostatic energy change for entire molecule
        double newNonBondedEnergy = getNonBondedEnergy(newConformation);
        energyChange += (newNonBondedEnergy - currentNonBondedEnergy);

        // calculate new solvation energy        
        /*double solvationEnergy = 0.0;
        List<Double> SASAlist = null;
        try { SASAlist = new DCLMAreaCalculator(0.0).calculateSASA(newPeptide); }
        catch (Exception e) { e.printStackTrace(); SASAlist = ShrakeRupleyCalculator.INSTANCE.calculateSASA(newPeptide); }
        for (int i=0; i < SASAlist.size(); i++)
        {
            double surfaceArea = SASAlist.get(i);
            double surfaceTension = peptide.contents.get(i).surfaceTension;
            double energy = surfaceArea * surfaceTension;
            //System.out.printf("%3d  %8.2f  %8.2f\n", i+1, surfaceArea, energy);
            solvationEnergy += energy;
        } */
        double newSolvationEnergy = 0.0; // solvationEnergy;

        // update energies and create new energy breakdown
        double newPotentialEnergy = currentConformation.energyBreakdown.potentialEnergy + energyChange;
        double newEnergy = newSolvationEnergy + newPotentialEnergy;
        EnergyBreakdown newEnergyBreakdown = new EnergyBreakdown(null, newEnergy, newSolvationEnergy, newPotentialEnergy, null, Forcefield.OPLS); 
        newConformation = currentConformation.setEnergyBreakdown(newEnergyBreakdown);

        // update conformations 
        previousConformation = currentConformation;
        previousNonBondedEnergy = currentNonBondedEnergy;
        currentConformation = newConformation;
        currentNonBondedEnergy = newNonBondedEnergy; 
        
        return newEnergy;
    }
    
    /** A method that returns to the state before the last mutation. 
    * This is useful because we can calculate an energy for a potential Monte Carlo move, reject the change, and then revert to the state before the change.
    */
    public void undoMutation()
    {
        if (previousConformation == null)
            throw new IllegalArgumentException("This method cannot be called in succession without a mutation in between calls");

        // return to previous state bu changing the current state to the previous state
        currentConformation = previousConformation;
        currentNonBondedEnergy = previousNonBondedEnergy;
        previousConformation = null;
        previousNonBondedEnergy = 0.0;

    }

    /**
     * Computes the OPLS non-bonded interaction energy between two sets of atoms.
     * The non-bonded interactions include van der Waals interactions and electrostatic interactions.
     * The two sets of atoms are the segments of a dihedral whose interactions are changing during a mutation.
     * The OPLS forcefield includes a fudge factor or scaling factor for the one 1-4 interaction in a dihedral change. 
     * @param set1 the first set of atoms
     * @param set2 the second set of atoms
     * @param dihedralAtom1 the atom at one terminal end of the dihedral angle
     * @param dihedralAtom4 the atom at the other terminal end of the dihedral angle. The interaction between atom1-atom4 must be scaled down according to OPLS.
     * @return the interaction energy in kcal
     */
    private static double getNonBondedEnergy(Set<Atom> set1, Set<Atom> set2, Atom dihedralAtom1, Atom dihedralAtom4)
    {
        double energy = 0.0;
        for (Atom atom1 : set1)
            {
                Integer atomType1 = atom1.type2;
                Integer atomClass1 = getOPLSClass(atomType1);

                double charge1 = getCharge(atomType1);
                double vdw_distance1 = getVDWDistance(atomClass1);
                double vdw_depth1 = getVDWDepth(atomClass1);
                
                for (Atom atom2 : set2)
                    {
                        Integer atomType2 = atom2.type2;
                        Integer atomClass2 = getOPLSClass(atomType2);

                        double charge2 = getCharge(atomType2);
                        double vdw_distance2 = getVDWDistance(atomClass2);
                        double vdw_depth2 = getVDWDepth(atomClass2);

                        // calculate the coulombic energy
                        double distance = Vector3D.distance(atom1.position, atom2.position);
                        // avoid blowing up the energy
                        if ( distance < MIN_DISTANCE )
                            distance = MIN_DISTANCE;
                        double electrostatic = (charge1 * charge2 * COULOMB_CONSTANT) / (distance * distance);

                        // apply the combining rules for epsilon and sigma if necessary
                        double sigma = vdw_distance1;
                        double epsilon = vdw_depth1;

                        if ( atomClass1 != atomClass2 )
                            {
                                sigma = Math.sqrt(vdw_distance1 * vdw_distance2);
                                epsilon = Math.sqrt(vdw_depth1 * vdw_depth2);
                            }

                        // calculate the steric energy
                        double temp = Math.pow(sigma / distance, 6);
                        double steric = 4.0 * epsilon * temp * ( temp - 1.0 );

                        // return the result
                        double scaling = 1.0; // assumed to be 1 because all interactions are 1,4 or greater
                        if (atom1.equals(dihedralAtom1) && atom2.equals(dihedralAtom4))
                            scaling = 0.5;
                        energy += (electrostatic + steric) * scaling;
                    }
            }
        return energy;
    }
 
    /**
     * Computes the non-bonded OPLS energy in a molecule between all atom pairs.
     * @param molecule the molecule to analyze
     * @return the non-bonded energy calculated on the OPLS force field
     */
    public static double getNonBondedEnergy(Molecule molecule)
    {
        List<Interaction> interactions = getInteractions(molecule);
        double energy = 0.0;
        for (Interaction i : interactions)
        {
            energy += i.interactionEnergy;
        }
        return energy;
    }

    /**
     * Computes the non-bonded OPLS interactions in a molecule between all atom pairs.
     * @param molecule the molecule to analyze
     * @return list of all non-bonded interactions
     */
    public static List<Interaction> getInteractions(Molecule molecule)
    {
        // get fields
        List<Atom> contents = molecule.contents;
        SimpleWeightedGraph<Atom,DefaultWeightedEdge> connectivity = molecule.connectivity;

        // generate list of valid atom pairs
        int estimatedSize = contents.size() * (contents.size()-1) / 2;
        List<Interaction> interactions = new ArrayList<Interaction>(estimatedSize);
        for (int i=0; i < contents.size(); i++)
            {
                Atom atom1 = contents.get(i);
                Integer atomType1 = atom1.type2;
                Integer atomClass1 = getOPLSClass(atomType1);

                double charge1 = getCharge(atomType1);
                double vdw_distance1 = getVDWDistance(atomClass1);
                double vdw_depth1 = getVDWDepth(atomClass1);

                for (int j=i+1; j < contents.size(); j++)
                    {
                        Atom atom2 = contents.get(j);
                        Integer atomType2 = atom2.type2;
                        Integer atomClass2 = getOPLSClass(atomType2);

                        double charge2 = getCharge(atomType2);
                        double vdw_distance2 = getVDWDistance(atomClass2);
                        double vdw_depth2 = getVDWDepth(atomClass2);

                        // calculate the graph-theoretic distance
                        DijkstraShortestPath<Atom,DefaultWeightedEdge> path = new DijkstraShortestPath<>(connectivity, atom1, atom2, 3.0);
                        List<DefaultWeightedEdge> pathEdges = path.getPathEdgeList();

                        // scale 1,4-interactions by 50%
                        double scaling = 1.0;
                        if ( pathEdges != null && pathEdges.size() < 3 )
                            continue;
                        else if ( pathEdges != null && pathEdges.size() == 3 )
                            scaling = 0.5;

                        // calculate the coulombic energy
                        double distance = Vector3D.distance(atom1.position, atom2.position);
                        if ( distance < MIN_DISTANCE )
                            distance = MIN_DISTANCE;
                        double electrostatic = (charge1 * charge2 * COULOMB_CONSTANT) / (distance * distance);

                        // apply the combining rules for epsilon and sigma if necessary
                        double sigma = vdw_distance1;
                        double epsilon = vdw_depth1;

                        if ( atomClass1 != atomClass2 )
                            {
                                sigma = Math.sqrt(vdw_distance1 * vdw_distance2);
                                epsilon = Math.sqrt(vdw_depth1 * vdw_depth2);
                            }

                        // calculate the steric energy
                        double temp = Math.pow(sigma / distance, 6);
                        double steric = 4.0 * epsilon * temp * ( temp - 1.0 );

                        // apply scaling
                        double energy = ( electrostatic + steric ) * scaling;
                        
                        // create the interaction
                        Set<Atom> atomList = ImmutableSet.of(atom1, atom2);
                        //Interaction interaction = new Interaction(atomList, energy, description);
                        Interaction interaction = new Interaction(atomList, energy);
                        interactions.add(interaction);
                    }
            }
        
        // return the result
        return ImmutableList.copyOf(interactions);
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

    public static void main(String[] args)
    {
        // Create peptide
        DatabaseLoader.go();
        List<ProtoAminoAcid> sequence = ProtoAminoAcidDatabase.getSpecificSequence("arg","met","standard_ala","gly","d_proline", "gly", "phe", "val", "hd", "l_pro");
        Peptide peptide = PeptideFactory.createPeptide(sequence);
        
        // Create OPLS calculator
        FixedSequenceOPLScalculator calculator = new FixedSequenceOPLScalculator(peptide);

        // Make a mutation
        // Pick a random residue
        Residue r = peptide.sequence.get(2);
        ProtoTorsion phi = r.phi;
        ProtoTorsion psi = r.psi;
        // only mutate one value
        double samePhiValue = phi.getDihedralAngle();
        double newPsiValue = 175.0;
        
        Peptide newPeptide = BackboneMutator.setPhiPsi(peptide, r, samePhiValue, newPsiValue);
        double calculatorPotentialEnergy = calculator.modifyDihedral(psi, newPsiValue, newPeptide);

        // Call Tinker on mutated peptide
        TinkerAnalysisJob tinkerAnalysisJob = new TinkerAnalysisJob(newPeptide, Forcefield.OPLS);
        TinkerAnalysisJob.TinkerAnalysisResult result = tinkerAnalysisJob.call();
        TinkerAnalyzeOutputFile outputFile = result.tinkerAnalysisFile;
        double tinkerPotentialEnergy = outputFile.totalEnergy;

        // Compare energy from Tinker with energy from OPLS calculator
        if (calculatorPotentialEnergy == tinkerPotentialEnergy)
            System.out.println("Success");
        else
            System.out.println("The tinker energy is : "  + tinkerPotentialEnergy + " and the calculator PE is : " + calculatorPotentialEnergy);
    }
}
