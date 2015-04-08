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
 * This class is designed for use in fixed sequence Monte Carlo simulations with mutations involving mutations at a reisdue's backbone angle and rotamer packing. 
 * It is meant for use in the Fragment Generation Monte Carlo
 * The OPLS force field is developed in <a href="http://pubs.acs.org/doi/abs/10.1021/ja9621760">Jorgensen et al., 1996</a>. */
public class FragmentGenerationOPLScalculator extends FixedSequenceOPLScalculator
{
    
    /** The total of all the torsional energy terms for the current conformation */ 
    public double currentSidechainTorsionalEnergy;

    /** The total of all the torsional energy terms for the previous conformation */
    public double previousSidechainTorsionalEnergy; 

    /** This constructor finds the energy of the sidechain torsions (all chis) in a peptide.
     * This will allow for rapid computation of the energy difference between confromers in the Monte Carlo */
    public FragmentGenerationOPLScalculator(Peptide startingPeptide)
    {
        super(startingPeptide);
        
        // Calculate total torsional energy 
        double tempTorsionalEnergy = 0.0;
        for (Residue r : startingPeptide.sequence)
        {   
           for (ProtoTorsion chi : r.chis)
               tempTorsionalEnergy += getDihedralEnergy(chi);
        }

        this.currentSidechainTorsionalEnergy = tempTorsionalEnergy;
        this.previousSidechainTorsionalEnergy = 0.0;
    }

    /** A method that returns to the state before the last mutation. 
    * This is useful because we can calculate an energy for a potential Monte Carlo move, reject the change, and then revert to the state before the change.
    */
    public void undoMutation()
    {
        super.undoMutation();
        currentSidechainTorsionalEnergy = previousSidechainTorsionalEnergy;
    }
 
    /** Calculates the energy of a new conformation following a residue's backbone angles being mutated and rotamer packing.
     * This method adds the change in energy of dihedrals and nonbonded interactions.
     * @param newConformation the mutated conformation whose energy will be calculated 
     * @param mutatedResidue the residue whose backbone angles are being changed
     * @return the energy of the new conformation in the OPLS force field 
     */
    public double calculateEnergy(Residue mutatedResidue, Peptide newConformation)
    {
        double energyChange = 0.0;
        // Find energy change in dihedrals of mutatedResidue
        Residue previousResidue = currentConformation.sequence.get(newConformation.sequence.indexOf(mutatedResidue));
        ProtoTorsion newPhi = mutatedResidue.phi;     
        ProtoTorsion oldPhi = previousResidue.phi;
        energyChange += (getDihedralEnergy(newPhi) - getDihedralEnergy(oldPhi));

        ProtoTorsion newPsi = mutatedResidue.psi;
        ProtoTorsion oldPsi = previousResidue.psi;
        energyChange += (getDihedralEnergy(newPsi) - getDihedralEnergy(oldPsi));

        ProtoTorsion newOmega = mutatedResidue.omega;
        ProtoTorsion oldOmega = previousResidue.omega;
        energyChange += (getDihedralEnergy(newOmega) - getDihedralEnergy(oldOmega));

        // Find energy change in all chis that are changed as a result of rotamer packing
        double newSidechainTorsionalEnergy = 0.0;
        for (Residue r : newConformation.sequence)
        {
            for (ProtoTorsion chi : r.chis)
                newSidechainTorsionalEnergy += getDihedralEnergy(chi);
        }
        energyChange += (newSidechainTorsionalEnergy - currentSidechainTorsionalEnergy);

        // Recalculate the non bonded energy
        double newNonBondedEnergy = getNonBondedEnergy(newConformation);
        energyChange += (newNonBondedEnergy - currentNonBondedEnergy);

        // Reset conformations and return new energy
        double oldEnergy = currentConformation.energyBreakdown.totalEnergy;
        double newEnergy = oldEnergy + energyChange;
        
        // Address solvation energy
        newConformation = newConformation.setEnergyBreakdown(new EnergyBreakdown(null, newEnergy, 0.0, newEnergy, null, Forcefield.OPLS));  
        
        // update conformations 
        previousConformation = currentConformation;
        previousNonBondedEnergy = currentNonBondedEnergy;
        currentConformation = newConformation;
        currentNonBondedEnergy = newNonBondedEnergy; 
        // update torsional energy
        previousSidechainTorsionalEnergy = currentSidechainTorsionalEnergy;
        currentSidechainTorsionalEnergy = newSidechainTorsionalEnergy;
        
        return newEnergy;
    }
    
    public static void main(String[] args)
    {
        // Create peptide
        DatabaseLoader.go();
        List<ProtoAminoAcid> sequence = ProtoAminoAcidDatabase.getSpecificSequence("arg","met","standard_ala","gly","d_proline", "gly", "phe", "val", "hd", "l_pro");
        Peptide peptide = PeptideFactory.createPeptide(sequence);
        
        // Create OPLS calculator
        FragmentGenerationOPLScalculator calculator = new FragmentGenerationOPLScalculator(peptide);

        // Make a mutation
        // Pick a random residue and change omega, phi, psi
        int residueNumber = 2;
        peptide = BackboneMutator.mutateOmega(peptide, residueNumber);
        Peptide newPeptide = BackboneMutator.mutatePhiPsi(peptide, residueNumber);
       
        // Change chis -- rotamer pack (to add)

        double calculatorPotentialEnergy = calculator.calculateEnergy(newPeptide.sequence.get(residueNumber), newPeptide);

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
  
