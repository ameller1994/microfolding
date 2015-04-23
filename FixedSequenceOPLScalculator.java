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

    /** electrostatic term that allows for quick computation of Coloumbic interaction. It is defined as k * q1 * q2 */
    private double[][] electrostaticMultiple;

    /** grouped term for VDW nonbonded energy term calculation. It is equal to epsilon * 4 * scaling. */
    private double[][] VDWMultiple; 

    /** sigma term for each pair of atoms in the peptide corresponding to this caluclator. Prevents look up each time */
    private double[][] sigmas;
    
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
        TinkerAnalysisJob tinkerAnalysisJob = new TinkerAnalysisJob(startingPeptide, Forcefield.OPLS);
        TinkerAnalysisJob.TinkerAnalysisResult result = tinkerAnalysisJob.call();
        TinkerAnalyzeOutputFile outputFile = result.tinkerAnalysisFile;
        double tinkerPotentialEnergy = outputFile.totalEnergy;
        EnergyBreakdown energyBreakdown = new EnergyBreakdown(null, outputFile.totalEnergy, 0.0, outputFile.totalEnergy, null, Forcefield.OPLS);

        // Populate arrays to speed up nonbonded energy calculation
        int totalAtoms = startingPeptide.contents.size();
        double[][] tempElectrostaticMultiple = new double[totalAtoms][totalAtoms];
        double[][] tempVDWMultiple = new double[totalAtoms][totalAtoms];
        double[][] tempSigmas = new double[totalAtoms][totalAtoms];
        
        // Loop through all atom pairs
        for (int i = 0; i < totalAtoms - 1; i++)
        {
            // Get atom type and class of atom1 
            Integer atomType1 = startingPeptide.contents.get(i).type2;
            Integer atomClass1 = getOPLSClass(atomType1);
            double charge1 = getCharge(atomType1);
            double vdw_distance1 = getVDWDistance(atomClass1);
            double vdw_depth1 = getVDWDepth(atomClass1);
            
            /*
            // designate size of array for this row
            tempElectrostaticMultiple[i] = new double[totalAtoms - i];
            tempVDWMultiple[i] = new double[totalAtoms - i]; 
            tempSigmas[i] = new double[totalAtoms - i];
            */

            for (int j = i+1; j < totalAtoms; j++)
            {
                // Get atom type and class of atom2
                Integer atomType2 = startingPeptide.contents.get(j).type2;
                Integer atomClass2 = getOPLSClass(atomType2);
                double charge2 = getCharge(atomType2);
                double vdw_distance2 = getVDWDistance(atomClass2);
                double vdw_depth2 = getVDWDepth(atomClass2);
               
                // calculate the graph-theoretic distance
                Atom atom1 = startingPeptide.contents.get(i);
                Atom atom2 = startingPeptide.contents.get(j);
                DijkstraShortestPath<Atom,DefaultWeightedEdge> path = new DijkstraShortestPath<>(startingPeptide.connectivity, atom1, atom2, 3.0);
                List<DefaultWeightedEdge> pathEdges = path.getPathEdgeList();

                // scale 1,4-interactions by 50%
                double scaling = 1.0;
                if ( pathEdges != null && pathEdges.size() < 3 )
                {
                    tempVDWMultiple[i][j] = 0.0;
                    tempSigmas[i][j] = 0.0;
                    tempElectrostaticMultiple[i][j] = 0.0;
                    continue;
                }
                else if ( pathEdges != null && pathEdges.size() == 3 )
                    scaling = 0.5;
    
                double sigma = vdw_distance1;
                double epsilon = vdw_depth1;

                if ( atomClass1 != atomClass2)
                {
                    sigma = Math.sqrt(vdw_distance1 * vdw_distance2);
                    epsilon = Math.sqrt(vdw_depth1 * vdw_depth2);
                }

                // populate arrays with pre-calculated values
                tempVDWMultiple[i][j] = 4.0 * scaling * epsilon;
                tempSigmas[i][j] = sigma;
                tempElectrostaticMultiple[i][j] = scaling * charge1 * charge2 * COULOMB_CONSTANT; 
            }
        }
        this.VDWMultiple = tempVDWMultiple;
        this.sigmas = tempSigmas;
        this.electrostaticMultiple = tempElectrostaticMultiple;

        this.currentConformation = startingPeptide.setEnergyBreakdown(energyBreakdown);
        this.currentNonBondedEnergy = getNonBondedEnergy(startingPeptide);
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
    /*
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
        }
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
    */

    /** Returns all dihedral indices that would change with the given dihedral that is undergoing a mutation
     * This method relies on the intuition that if one changes a dihedral a1-a2-a3-a4 then all dihedrals with the same a2 and a3 are also changing.
     * This can be shown by doing a single torsional mutation and looking at the overall change in torsional energy in Tinker.
     * @param a dihedral that is being mutated
     * @param peptide the corresponding conformation for the dihedral
     * @return a set containing all dihedrals that are changing as a result of this dihedral mutation in lists of indices 
     */
    public static Set<List<Integer>> getDihedralChanges(ProtoTorsion mutatedDihedral, Peptide peptide)
    {
        // Get adjacent atoms not including a2 or a3 to make combinations of a-a2-a3-a
        Set<Atom> connectedToAtom2 = peptide.getAdjacentAtoms(mutatedDihedral.atom2);
        connectedToAtom2.remove(mutatedDihedral.atom3);

        Set<Atom> connectedToAtom3 = peptide.getAdjacentAtoms(mutatedDihedral.atom3);
        connectedToAtom3.remove(mutatedDihedral.atom2);
        
        Set<List<Integer>> returnSet = new HashSet<>();
        for (Atom a1 : connectedToAtom2)
        {
            for (Atom a4 : connectedToAtom3)
            {
                List<Integer> torsionIndices = new LinkedList<>();
                torsionIndices.add(peptide.contents.indexOf(a1));
                torsionIndices.add(peptide.contents.indexOf(mutatedDihedral.atom2));
                torsionIndices.add(peptide.contents.indexOf(mutatedDihedral.atom3));
                torsionIndices.add(peptide.contents.indexOf(a4));
                returnSet.add(torsionIndices);
            }
        }
        return returnSet;
    }

    /** Returns the energy of a dihedral angle 
     * This method queries the OPLS forcefield and calculates the dihedral energy based on the OPLS force field.
     * This method uses the formula E = sigma( Vi/2*(1+/-cos(Per_i*(phi - Phase_i))) ) where V is the amplitude, Per is the periodicity.
     * @param protoTorsion the proto torsion of interest
     * @retun the energy value in the OPLS molecular mechanics force field energy calculation
     */
    public static double getDihedralEnergy(ProtoTorsion protoTorsion)
    {
        OPLSforcefield.TorsionalParameter torsionalParameter = getTorsionalParameter(protoTorsion);
        
        // Perform calculation of energy change using the formula: E = sigma( Vi/2*(1+/-cos(Per_i*(phi - Phase_i)) ))
        // where V is the amplitude, Per is the periodicity.
        double angle = protoTorsion.getDihedralAngle();
        return getDihedralEnergy(torsionalParameter, angle); 

    }

    public static double getDihedralEnergy(List<Integer> indices, Peptide conformation)
    {
        List<Integer> atomClasses = new LinkedList<>();
        atomClasses.add(getOPLSClass(conformation.contents.get(indices.get(0)).type2));
        atomClasses.add(getOPLSClass(conformation.contents.get(indices.get(1)).type2));
        atomClasses.add(getOPLSClass(conformation.contents.get(indices.get(2)).type2));
        atomClasses.add(getOPLSClass(conformation.contents.get(indices.get(3)).type2));
        
        OPLSforcefield.TorsionalParameter torsionalParameter = getTorsionalParameter(atomClasses);
        
        Vector3D v1 = conformation.contents.get(indices.get(0)).position; 
        Vector3D v2 = conformation.contents.get(indices.get(1)).position; 
        Vector3D v3 = conformation.contents.get(indices.get(2)).position; 
        Vector3D v4 = conformation.contents.get(indices.get(3)).position;

        double angle = AbstractTorsion.getDihedralAngle(v1,v2,v3,v4);
        return getDihedralEnergy(torsionalParameter, angle);
    }

    
    /** This method finds the energy of a dihedral given an angle measurement and a torsional parameter 
     * @param torsionalParameter the torsional parameter for the given angle measurement
     * @param angle the angle in degrees
     * @return energy of dihedral in kcal
     */
    private static double getDihedralEnergy(OPLSforcefield.TorsionalParameter torsionalParameter, double angle)
    {
        double angleInRadians = Math.toRadians(angle);
        double[] amplitude = torsionalParameter.amplitudes;

        double dihedralEnergy = amplitude[0] / 2 * (1 + Math.cos(angleInRadians));
        // Phase for even terms is 180 degrees or Math.PI
        dihedralEnergy += amplitude[1] / 2 * (1 - Math.cos(2*(angleInRadians - Math.PI)));
        dihedralEnergy += amplitude[2] / 2 * (1 + Math.cos(3*angleInRadians));

        return dihedralEnergy;
    }

    public static double getAngleEnergy(List<Integer> indices, Peptide conformation)
    {
        // Query force field for corresponding parameter
        List<Integer> atomClasses = new LinkedList<>();
        atomClasses.add(getOPLSClass(conformation.contents.get(indices.get(0)).type2));
        atomClasses.add(getOPLSClass(conformation.contents.get(indices.get(1)).type2));
        atomClasses.add(getOPLSClass(conformation.contents.get(indices.get(2)).type2));
        
        OPLSforcefield.AngleParameter angleParameter = getAngleParameter(atomClasses);

        // Find angle between atoms corresponding to input indices
        Atom a1 = conformation.contents.get(indices.get(0));
        Atom a2 = conformation.contents.get(indices.get(1));
        Atom a3 = conformation.contents.get(indices.get(2));
        double angle = Molecule.getAngle(a1, a2, a3);

        return getAngleEnergy(angleParameter, angle);
    }

    private static OPLSforcefield.AngleParameter getAngleParameter(List<Integer> atomClasses)
    {
        OPLSforcefield.AngleParameter angleParameter = OPLSforcefield.ANGLE_MAP.get(atomClasses);
        if (angleParameter == null)
        {
            // Reverse atom class list and query again
            List<Integer> atomClassesReversed = Lists.reverse(atomClasses);
            angleParameter = OPLSforcefield.ANGLE_MAP.get(atomClassesReversed);
            if (angleParameter == null)
                throw new NullPointerException("Undefined angle parameter for classes: " + atomClasses.toString());
        }
        return angleParameter;
    }

    private static double getAngleEnergy(OPLSforcefield.AngleParameter angleParameter, double angle)
    {
        return  angleParameter.k_0 * Math.pow(angle - angleParameter.theta_0, 2);
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
                        double electrostatic = (charge1 * charge2 * COULOMB_CONSTANT) / (distance);

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
                        energy += (electrostatic * + steric) * scaling;
                    }
            }
        return energy;
    }
 
    /**
     * Computes the non-bonded OPLS energy in a molecule between all atom pairs.
     * @param conformation the molecule to analyze
     * @return the non-bonded energy calculated on the OPLS force field
     */
    public double getNonBondedEnergy(Peptide conformation)
    {
        int totalAtoms = conformation.contents.size();
        double nonBondedEnergy = 0.0;
        for (int i = 0; i < totalAtoms - 1; i++)
        {
            for (int j = i+1; j < totalAtoms; j++)
            {
                Atom atom1 = conformation.contents.get(i);
                Atom atom2 = conformation.contents.get(j);
                double distance = Vector3D.distance(atom1.position, atom2.position);
                
                // avoid blowing up the energy
                if ( distance < MIN_DISTANCE )
                    distance = MIN_DISTANCE;
                
                double electrostatic = electrostaticMultiple[i][j] / distance; 
                double temp = Math.pow(sigmas[i][j] / distance, 6);
                double steric = VDWMultiple[i][j] * temp * (temp - 1);
                
                //Debugging code
                /*if (electrostatic != 0.0)
                    System.out.println("Electrostatic " + conformation.getAtomString(atom1) +  " - " + conformation.getAtomString(atom2) + " D: " + distance +  " E:  " + electrostatic);
                if (electrostatic !=  0.0)
                    numberInteractions++; 
                */

                // scaling is already included in the electrostatic and steric terms
                nonBondedEnergy += (electrostatic +  steric);
            }
        }

        return nonBondedEnergy; 
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
                        double electrostatic = (charge1 * charge2 * COULOMB_CONSTANT) / (distance);

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
     * Gets the torsional parameter corresponding to the input atom classes. 
     * This method ensures that proto torsions with ac1-ac2-ac3-ac4 are also searched in the opposite order of atom classes.
     * @param protoTorsion a proto torsion whose torsional parameters will be queried
     * @return a OPLSforcefield TorsionalParameter object containing the amplitude and periodicity corresponding to the torsion
    */
    public static OPLSforcefield.TorsionalParameter getTorsionalParameter(ProtoTorsion protoTorsion)
    {
        // Get atom classes for the proto torsion
        List<Integer> atomClasses = new LinkedList<>();
        atomClasses.add(getOPLSClass(protoTorsion.atom1.type2));
        atomClasses.add(getOPLSClass(protoTorsion.atom2.type2));
        atomClasses.add(getOPLSClass(protoTorsion.atom3.type2));
        atomClasses.add(getOPLSClass(protoTorsion.atom4.type2));
        
        return getTorsionalParameter(atomClasses);    
    }

    public static OPLSforcefield.TorsionalParameter getTorsionalParameter(List<Integer> atomClasses)
    {
        OPLSforcefield.TorsionalParameter torsionalParameter = OPLSforcefield.TORSIONAL_MAP.get(atomClasses);
        if (torsionalParameter == null)
        {
            // Reverse atom class list and query again
            List<Integer> atomClassesReversed = Lists.reverse(atomClasses);
            torsionalParameter = OPLSforcefield.TORSIONAL_MAP.get(atomClassesReversed);
            if (torsionalParameter == null)
                throw new NullPointerException("Undefined torsional parameter for classes: " + atomClasses.toString());
        }
        return torsionalParameter;
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

   }
}
