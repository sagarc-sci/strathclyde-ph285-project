import numpy, scipy, pandas
import itertools, logging, sys
from collections.abc import Callable
from functools import cache


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
if not LOG.hasHandlers():
    LOG.addHandler(logging.StreamHandler(sys.stderr))


class Particle(object):
    '''
        Model a charged particle that can interact with photons
        
        Interactions are modelled by adding transitions that can move the particle
        to a different excitation or ionisation
    '''

    def __init__(self, sybmol: str, protons: int, quantum_number: int, charge: int):
        self.symbol: str = sybmol
        self.protons: int = protons
        self.quantum_number: int = quantum_number
        self.charge: int = charge
        self.outgoing_transitions: set['Transition'] = set()
        self.incoming_transitions: set['Transition'] = set()
    
    def __repr__(self) -> str:
        return f'[{self.symbol}]'

    def add_outgoing_transition(self, transition: 'Transition') -> 'Particle':
        # Automatically invoked when a Transition object is instantiated
        self.outgoing_transitions.add(transition)
        return self

    def add_incoming_transition(self, transition: 'Transition') -> 'Particle':
        # Automatically invoked when a Transition object is instantiated
        self.incoming_transitions.add(transition)
        return self
    
    @cache
    def ground_state(self) -> 'Particle':
        '''
            Traverses the modelled transitions to find the ground state (i.e. quantum number equals 1)
            of this particle

            Does not create a new ground state particle if one isn't already modelled in the transition graph
            e.g. if this particle is [H-II] and [H-I] isn't modelled then the method returns None
        '''

        if 1 == self.quantum_number:
            return self
        else:
            excitation = self
            while (excitation := excitation.previous_excitation()):
                if 1 == excitation.quantum_number:
                    return excitation
        return None
    
    @cache
    def previous_excitation(self) -> 'Particle':
        '''
            Traverses the modelled transitions to find the previous excitation state of this particle
            (i.e. quantum number immediately less than that of this partcile)

            Does not create a new particle if one isn't already modelled in the transition graph
            e.g. if this particle is [H-III] and [H-II] isn't modelled but [H-I] is modelled
            then this method returns [H-I]
        '''

        return max((transition.source for transition in self.incoming_transitions if isinstance(transition, BoundBoundAbsorption)),
                   key=lambda particle: particle.quantum_number, default=None)

    @cache
    def next_excitation(self) -> 'Particle':
        '''
            Traverses the modelled transitions to find the next excitation state of this particle
            (i.e. quantum number immediately greater than that of this partcile)

            Does not create a new particle if one isn't already modelled in the transition graph
            e.g. if this particle is [H-I] and [H-II] isn't modelled but [H-III] is modelled
            then this method returns [H-III]
        '''

        return min((transition.destination for transition in self.outgoing_transitions if isinstance(transition, BoundBoundAbsorption)),
                   key=lambda particle: particle.quantum_number, default=None)
    
    @cache
    def elemental_state(self) -> 'Particle':
        '''
            Finds the elemental state of this particle (i.e. net charge of zero)

            Doesn't create a new elemental particle if one isn't modelled in the transition graph
            e.g. if this particle is [H+] and [H] isn't modelled then this method returns None
        '''

        if 0 == self.charge:
            return self
        elif self.charge > 0:
            next_function = Particle.previous_ion
        else:
            next_function = Particle.next_ion
        ion = self
        while (ion := next_function(ion)):
            if 0 == ion.charge:
                return ion
        return None

    @cache
    def previous_ion(self) -> 'Particle':
        '''
            Finds the previous ionisation state of this particle
            (i.e. net charge immediately less than that of this particle)

            Doesn't create a new particle if one isn't modelled in the transition graph
            e.g. if this particle is [He+2] and [He+1] isn't modelled but [He] is modelled then
            this method returns [He]
        '''

        for transition in self.incoming_transitions:
            if isinstance(transition, BoundFreeAbsorption):
                return transition.source
        return None
    
    @cache
    def next_ion(self) -> 'Particle':
        '''
            Finds the next ionisation state of this particle
            (i.e. net charge immediately greater than that of this particle)

            Doesn't create a new particle if one isn't modelled in the transition graph
            e.g. if this particle is [He] and [He+1] isn't modelled but [He+2] is modelled then
            this method returns [He+2]
        '''

        for transition in self.outgoing_transitions:
            if isinstance(transition, BoundFreeAbsorption):
                return transition.destination
        return None
    
    @cache
    def degeneracy(self) -> float:
        '''
            Returns degeneracy of quantum number describing the current state

            For Hydrogenic species with a single electron described using quantum number n,
            degeneracy is 2 * n^2
        '''

        # Only true for Hydrogenic species
        return 2 * self.quantum_number ** 2
    
    @cache
    def ionisation_energy(self) -> float:
        '''
            Returns ionisation energy of this particle

            This is the same as negative of electron energy state
            For Hydrogenic species with single electron,
            ionisation energy, X = R_inf * (Z^2 / n^2)
        '''

        # Only true for Hydrogenic species
        return scipy.constants.physical_constants['Rydberg constant times hc in J'][0] * (self.protons / self.quantum_number) ** 2

    @cache
    def ionisation_wavelength(self) -> float:
        '''
            Returns the ionisation wavelength of this particle

            Ionisation wavelength is hc/X where X is the ionisation energy
        '''

        return (scipy.constants.h * scipy.constants.c) / self.ionisation_energy()
    
    @cache
    def partition_function(self, temperature: float) -> float:
        '''
            Returns the contribution of this particle's excitation state to partition function
            for use in Saha and Boltzmann equations

            Partition function is defined as Z = sum(g_i * exp(-(E_i - E1)/kT))
            where g_i is the degeneracy of the energy level E_i; E1 being ground state energy

            See
              Carroll, B. W., & Ostlie, D. A. 2018, An introduction to modern astrophysics, p. 214
              (Second edition; Cambridge: Cambridge University Press)
        '''

        if self.protons - self.charge == 0:
            return 1.
        return self.degeneracy() * numpy.exp((self.ionisation_energy() - self.ground_state().ionisation_energy())
                                             / (scipy.constants.k * temperature))


class Transition(object):
    '''
        Models an interaction between photon and a charged particle
        that alters the particle and/or photon states

        This model provides the `opacity` method to obtain transition probabilities
        using the `cross_section` method implemented by subclasses
    '''

    def __init__(self, source: 'Particle', destination: 'Particle'):
        self.source: 'Particle' = source
        self.destination: 'Particle' = destination

        self.source.add_outgoing_transition(self)
        self.destination.add_incoming_transition(self)
    
    def symbol(self) -> str:
        '''
            Label for transition for use in printing
        '''

        return NotImplemented
    
    def __repr__(self) -> str:
        return f'{self.source} -- {self.symbol()} --> {self.destination}'
    
    def cross_section(self, wavelength: float|numpy.ndarray, cell: 'Cell') -> float|numpy.ndarray:
        '''
            Analogue for cross-sectional area (m^2) of an interaction modelled by this transition

            e.g. expected number of interactions divided by flux of modelled particles through an area
        '''

        return NotImplemented
    
    def opacity(self, wavelength: float|numpy.ndarray, cell: 'Cell') -> float|numpy.ndarray:
        '''
            Returns the probability of interaction defined by this transition occuring
            per unit length occupied by the modelled particles (m^-1)
        '''

        return cell.particle_number_densities[self.source] * self.cross_section(wavelength, cell)


class ThomsonScattering(Transition):
    '''
        Models Thomson scattering off of an electron
    '''

    def symbol(self):
        return 'TH'
    
    def cross_section(self, wavelength: float|numpy.ndarray, cell: 'Cell') -> float|numpy.ndarray:
        '''
            Cross section of Thomson scattering off of an electron;
            has the same value at all wavelengths

            See
              Carroll, B. W., & Ostlie, D. A. 2018, An introduction to modern astrophysics, p. 246
              (Second edition; Cambridge: Cambridge University Press)
        '''

        return 6.65e-29 # m2


class FreeFreeAbsorption(Transition):
    '''
        Models interaction of a photon with a free electron in the potential well of a nearby ion

        See
          Carroll, B. W., & Ostlie, D. A. 2018, An introduction to modern astrophysics, p. 246
          (Second edition; Cambridge: Cambridge University Press)
    '''

    CONSTANT_TERM: float = ((numpy.sqrt(32 * numpy.pi) * scipy.constants.e ** 6)
                     / ((4 * numpy.pi * scipy.constants.epsilon_0)
                        * (3 * numpy.sqrt(3) * scipy.constants.h * scipy.constants.c ** 4)
                        * numpy.sqrt(scipy.constants.k * scipy.constants.m_e ** 3)))

    def symbol(self) -> str:
        return 'FF'

    def gaunt_factor(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        # Quantum mechanical correction factor
        # Not implemented
        return 1.

    def cross_section(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        '''
            Wavelength dependendent cross section of free-free absorption of a photon by a free electron

            For Hydrogenic species this is given in SI units by

              sum(
                ((g_ff * 32 pi)^(1/2) * e^6 * Z_i^2 * n_i * lambda^3)
                  / (4 * pi * epsilon_0 * 3^(3/2) * c^4 * h * (k * m_e^3 * T)^(1/2))
              )

            where the sum is over all ions (i) with n_i, the number density of the ion

            See
              Shields, J. V., Kerzendorf, W., Smith, I. G., et al. 2025 (arXiv), http://arxiv.org/abs/2504.17762, Eq. (9)
        '''

        # Only true for Hydrogenic species
        return (self.CONSTANT_TERM
                * (wavelength ** 3 / numpy.sqrt(cell.temperature))
                * sum(particle_number_density * particle.protons ** 2
                      for particle, particle_number_density in cell.particle_number_densities.items()
                      if particle.charge != 0 and particle is not cell.atmosphere.electron)
                * self.gaunt_factor(wavelength, cell))


class BoundFreeAbsorption(Transition):
    '''
        Models ionisation of an atom or ion by a sufficiently energetic photon

        See
          Carroll, B. W., & Ostlie, D. A. 2018, An introduction to modern astrophysics, p. 245
          (Second edition; Cambridge: Cambridge University Press)
    '''

    def __init__(self, source: 'Particle', destination: 'Particle'):
        super().__init__(source, destination)
        self.constant_term: float = (((64 * (numpy.pi ** 4) * (scipy.constants.e ** 10) * scipy.constants.m_e)
                                      / (3 * numpy.sqrt(3) * (scipy.constants.c ** 4) * (scipy.constants.h ** 6)
                                         * ((4 * numpy.pi * scipy.constants.epsilon_0) ** 5)))
                                         * (self.source.protons ** 4 / self.source.quantum_number ** 5))

    def symbol(self) -> str:
        return 'BF'

    def gaunt_factor(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        # Quantum mechanical correction factor
        # Not implemented
        return 1.

    def cross_section(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        '''
            Wavelength dependent cross section of photoionisation of an atom or ion
            Cross section is 0 if the photon does not have sufficient energy (i.e. ionisation energy)

            For Hydrogenic species this is given in SI units by

              (g_bf * 64 * pi^4 * e^10 * m_e * Z^4 * lambda^3) / (4 * pi * epsilon_0 * 3^(3/2) * c^4 * h^6 * n^5)

            See
              Shields, J. V., Kerzendorf, W., Smith, I. G., et al. 2025 (arXiv), http://arxiv.org/abs/2504.17762, Eq. (8)
        '''

        # Only true for Hydrogenic species
        return numpy.where(wavelength <= self.source.ionisation_wavelength(),
                           self.constant_term * (wavelength ** 3) * self.gaunt_factor(wavelength, cell),
                           0.)


class BoundBoundAbsorption(Transition):
    '''
        Models absorption of a photon by a bound electron in an atom or ion
        to transition to a higher excitation state

        See
          Carroll, B. W., & Ostlie, D. A. 2018, An introduction to modern astrophysics, p. 245
          (Second edition; Cambridge: Cambridge University Press)
    '''

    def __init__(self, source: 'Particle', destination: 'Particle', oscillator_strength: float):
        super().__init__(source, destination)
        self.line_wavelength: float = 1 / ((1 / self.source.ionisation_wavelength()) - (1 / self.destination.ionisation_wavelength()))
        self.line_frequency: float = scipy.constants.c / self.line_wavelength
        self.cross_section_constant_term: float = ((scipy.constants.e ** 2 / (4
                                                                              * scipy.constants.epsilon_0
                                                                              * scipy.constants.m_e
                                                                              * scipy.constants.c))
                                                                              * oscillator_strength)
        self.doppler_width_constant_term: float = (numpy.sqrt(2 * scipy.constants.k / (scipy.constants.m_p * self.source.protons))
                                                   / self.line_wavelength)
                                 
    def symbol(self) -> str:
        return 'BB'
    
    def doppler_profile(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        '''
            Models widening of spectral lines due to Doppler shift of emitted (and consequently absorbed) wavelengths
            due to thermal and non-thermal motion of particles

            This implementation approximates the Doppler profile using Gaussian distribution centered around the
            line wavelength and standard deviation of FWHM obtained using mean velocity of particles (sqrt(2kT/m))
            (based on Maxwell-Boltzmann distribution)
        '''

        doppler_width = self.doppler_width_constant_term * numpy.sqrt(cell.temperature)
        frequency = scipy.constants.c / wavelength
        return numpy.exp(- ((frequency - self.line_frequency) / doppler_width) ** 2) / (doppler_width * numpy.sqrt(numpy.pi))
    
    def line_profile(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        '''
            Models widening of spectral lines due to various processes
            Only Doppler widening has been implemented

            See
              Carroll, B. W., & Ostlie, D. A. 2018, An introduction to modern astrophysics, p. 268-273
              (Second edition; Cambridge: Cambridge University Press)
        '''

        return self.doppler_profile(wavelength, cell)

    def gaunt_factor(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        # Quantum mechanical correction factor
        # Not implemented
        return 1.

    def cross_section(self, wavelength: float|numpy.ndarray, cell: 'DenseCell') -> float|numpy.ndarray:
        '''
            Wavelength dependent cross section of excitation transition of a bound electron in an atom or ion

            Excitation occurs when photon has a specific wavelength with some broadening due to uncertainty principle,
            motion of particles causing Doppler shift in observed wavelengths, etc. This widening is modelled by the
            line_profile method

            Additionally, each state transition has a probability given by oscillator strength for the orbitals involved
            and a correction term is required to account for stimulated emission

            In SI units the cross section is given by

              ((pi * e^2 * f_lu) / (4 * pi * epsilon_0 * m_e * c)) * correction_term * line_profile

            where f_lu is the oscillator strength for transition from lower orbital (l) to upper orbital (u)
            and correction term for stimulated emission given by (1 - (g_l * n_u) / (g_u * n_l))
            where g_i is the degeneracy of orbital i and n_i is the number density of particles in that state

            See
              Shields, J. V., Kerzendorf, W., Smith, I. G., et al. 2025 (arXiv), http://arxiv.org/abs/2504.17762, Eq. (11)
        '''

        return (self.cross_section_constant_term
                * (1 - ((self.source.degeneracy() * cell.particle_number_densities[self.destination])
                        / (self.destination.degeneracy() * cell.particle_number_densities[self.source])))
                * self.line_profile(wavelength, cell)
                * self.gaunt_factor(wavelength, cell))


class OscillatorStrength(object):
    '''
        Lookup table for oscillator strengths of orbital transitions

        See
          Menzel, D. H., & Pekeris, C. L. 1935, Monthly Notices of the Royal Astronomical Society, 96 (OUP), p. 77-110
          (Precalculated f-values for Hydrogen atom are taken from pages 82-84)
    '''

    def __init__(self, filename):
        self.dataset = pandas.read_csv(filename)

    def value(self, protons: int, charge: int, quantum_number: int, excited_quantum_number: int):
        query = ((self.dataset['protons'] == protons)
                 & (self.dataset['charge'] == charge)
                 & (self.dataset['quantum_number'] == quantum_number)
                 & (self.dataset['excited_quantum_number'] == excited_quantum_number))
        result = self.dataset.loc[query]
        if result.size > 0:
            return self.dataset.at[result.index[0], 'oscillator_strength']
        else:
            # Use Menzel-Pekeris approximation for Hydrogenic species
            #  g_bf * (32 / (pi * 3^(3/2))) * (1 / (l^5 * u^3)) * (1/l^2 - 1/u^2)^(-3)
            # where l and u are the quantum numbers of lower and upper orbitals
            # and g_bf is the gaunt factor

            # Quantum mechanical correction factor
            # Not implemented
            gaunt_factor = 1.

            # Only true for Hydrogenic species
            return (((32 / (3 * numpy.pi * numpy.sqrt(3))) / (quantum_number ** 5 * excited_quantum_number ** 3))
                    * ((1 / quantum_number ** 2) - (1 / excited_quantum_number ** 2)) ** (-3)
                    * gaunt_factor)


class Cell(object):
    '''
        An optical depth point where opacity is estimated for Monte-Carlo integration
    '''

    def opacity(self, wavelength: float|numpy.ndarray) -> float|numpy.ndarray:
        '''
            Estimate wavelength dependenty opacity based on all Photon-Particle
            interactions modelled in this Cell
        '''

        return NotImplemented
    
    def __mul__(self, other):
        '''
            Convinience method to invoke opacity on a vector of wavelengths
            without having to explicitly vectorise the Cell matrix

            e.g. numpy.ndarray[Cell] * numpy.ndarray[Wavelength] calculates
            opacity at wavelength in the corresponding cell (paired element-wise)
        '''
        if isinstance(other, (float, numpy.floating, numpy.ndarray)):
            return self.opacity(other)
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)


class EmptyCell(Cell):
    '''
        A completely transparent cell for convinience in non-planar geometries
        to ignore atmospheric effects on photons that travel shorter distances to
        escape the atmosphere
    '''

    def opacity(self, wavelength: float|numpy.ndarray) -> float|numpy.ndarray:
        '''
            Opacity of empty cell is always 0
        '''

        return 0.


class DenseCell(Cell):
    '''
        An atmospheric cell that estimates opacities due to Particle-Photon interactions
        modelled by the Transition graph

        This is instantiated by Atmosphere object with elemental composition, temperature and number density
        at a given optical depth point
    '''

    def __init__(self, atmosphere: 'Atmosphere', temperature: float, number_density: float, elemental_composition: dict['Particle', float]):
        self.atmosphere: 'Atmosphere' = atmosphere
        self.temperature: float = temperature
        self.number_density: float = number_density
        self.elemental_composition: dict['Particle', float] = elemental_composition
        self.particle_number_densities: dict['Particle', float] = self.solve_particle_number_densities()

    def solve_particle_number_densities(self) -> dict['Particle', float]:
        '''
            Solve for number densities of specific excited and ionised particles given the elemental composition
            using Saha ionisation equation and Boltzmann equation for excitation states;
            Number density is the number of particles in a unit volume (m^-3)

            e.g. if the atmosphere is 100% Hydrogen, this method estimates the number densities
            for specific excitation states (i.e. [H-I], [H-II], [H-III], etc.) and ionisation states
            (i.e. [H-], [H], [H+], etc.)

            See
              Carroll, B. W., & Ostlie, D. A. 2018, An introduction to modern astrophysics, p. 209-219
              (Second edition; Cambridge: Cambridge University Press)
        '''

        particle_number_densities: dict['Particle', float] = dict()

        ion_number_densities, electron_number_density = self.solve_ion_number_densities()

        particle_number_densities[self.atmosphere.electron] = electron_number_density

        for ion, ion_number_density in ion_number_densities.items():
            particle_number_densities.update(self.solve_excitation_number_densities(ion, ion_number_density))
        
        return particle_number_densities

    def solve_ion_number_densities(self, max_electron_density_error: float=0.01, max_iterations: int=1000) -> tuple[dict['Particle', float], float]:
        '''
            Solves for ion number densities for each element in atmosphere
            given the number density fraction of the element and total number density
            using the Saha ionisation equation

              N_i+1 = (N_i / N_e) * (2 * Z_i+1 / Z_i) * (2 * pi * m_e * k * T / h^2)^(3/2) * exp(- X_i / (k * T))
            
            where N_i is the number density of previous ionisation state (e.g. [H] for [H+]),
            N_e is the electron number density, X_i is the ionisation energy of previous ionisation state
            (i.e. energy to get to current state) and Z are the partition functions (see Particle.partition_function)

            Additionally,

              sum(N_i) = N_element = f_element * N
            
            f_element being the number density fraction of the element and N being the total number density

            Also assuming there are no other sources of free electrons in this cell,

              N_e = sum(charge_i * N_i)
            
            where charge_i is the charge of i-th ion (i.e. electrons lost by that ion)

            This method solves the system of equations to obtain number densities of ions and electron number density
        '''

        # Traverse the transition graph to collect ions of an element
        # Ions modelled in transition graph can be in any excitation state
        # always use the ground state ions
        ions_of_element: dict['Particle', list['Particle']] = dict()
        for element in self.elemental_composition:
            ions = set([element])
            for next_function in [Particle.previous_ion, Particle.next_ion]:
                ion = element
                while (ion := next_function(ion)):
                    ion = ion.ground_state()
                    ions.add(ion)
            # Sort the ions for sequential access
            ions_of_element[element] = sorted(ions, key=lambda ion: ion.charge)

        # Calculate the partition functions for each ion at the cell temperature
        partition_function_values: dict['Particle', float] = dict()
        for ions in ions_of_element.values():
            for ion in ions:
                partition_function_values[ion] = ion.partition_function(self.temperature)
                excitation = ion
                while(excitation := excitation.next_excitation()):
                    partition_function_values[ion] += excitation.partition_function(self.temperature)

        # Initialise number density of each ion to equally divide element's number density
        # i.e. N_ion = f_element * N / (number of ions of element)
        ion_number_densities: dict['Particle', float] = \
            dict((ion, self.elemental_composition[element] * self.number_density / len(ions))
                 for element, ions in ions_of_element.items()
                 for ion in ions)
        electron_number_density = 0.
        relative_error = 0.
        has_electron_density_converged = False
        constant_term = 2 * ((2 * numpy.pi * scipy.constants.m_e * scipy.constants.k) ** 1.5) / (scipy.constants.h ** 3)
        for iteration in range(max_iterations):
            # Assume the estimated ion number densities are correct and
            # calculate electron number density as N_e = sum(charge_i * N_i)
            updated_electron_number_density = sum(ion.charge * ion_number_density
                                                  for ion, ion_number_density in ion_number_densities.items())
            if 0 != electron_number_density:
                relative_error = numpy.abs(updated_electron_number_density - electron_number_density) / electron_number_density
            # If estimated ion number densities are correct
            # electron number densities converge in subsequent iterations
            if numpy.isclose(electron_number_density, updated_electron_number_density, rtol=max_electron_density_error):
                electron_number_density = updated_electron_number_density
                has_electron_density_converged = True
                break
            # Otherwise adjust ion number densities using the electron number density
            # and Saha ionisation equation
            electron_number_density = updated_electron_number_density
            for element, ions in ions_of_element.items():
                for i in range(1, len(ions)):
                    ion_number_densities[ions[i]] = ((constant_term
                     * (partition_function_values[ions[i]] / partition_function_values[ions[i - 1]])
                     * (self.temperature ** 1.5)
                     * numpy.exp(- ions[i - 1].ionisation_energy() / (scipy.constants.k * self.temperature)))
                     * (ion_number_densities[ions[i - 1]] / electron_number_density))
                # Scale the calculated ion number densities
                # so that the number density fraction of element still holds
                scaling_factor = (self.elemental_composition[element] * self.number_density
                                  / sum(ion_number_densities[ion] for ion in ions))
                for ion in ions:
                    ion_number_densities[ion] *= scaling_factor
        if not has_electron_density_converged:
            LOG.warning(f'Electron number density did not converge (relative error = {relative_error:0.2f})')

        return ion_number_densities, electron_number_density
    
    def solve_excitation_number_densities(self, particle: 'Particle', particle_number_density: float) -> dict['Particle', float]:
        '''
            Solves for number densities of excitation states for each ion
            in atmosphere using the Boltzmann equation for excitation states

              N_b = N_a * (g_b / g_a) * exp(- (E_b - E_a) / (k * T))
            
            where N_i are number densities of excitation states i
            and g_i are degeneracies of orbital described by quantum number i
            and E_i are energy levels (equal to - X_i; the ionisation energy)

            Additionally, sum(N_i) = N_ion
        '''

        # Calculate partition function values (these are relative to ground state)
        partition_function_values: dict['Particle', float] = dict()
        partition_function_values[particle] = particle.partition_function(self.temperature)
        excitation = particle
        while (excitation := excitation.next_excitation()):
            partition_function_values[excitation] = excitation.partition_function(self.temperature)

        # Use sum(N_i) = N_ion to rewrite Boltzmann equation using N_ion
        scaling_factor = particle_number_density / sum(partition_function_values.values())

        excitation_number_densities: dict['Particle', float] = \
            dict((excitation, partition_function_value * scaling_factor)
                 for excitation, partition_function_value in partition_function_values.items())
        
        return excitation_number_densities

    def opacity(self, wavelength: float|numpy.ndarray) -> float|numpy.ndarray:
        '''
            Opacity of this cell is sum of opacities due to modelled transitions
        '''

        return sum(transition.opacity(wavelength, self) for transition in self.atmosphere.transitions)


class Atmosphere(object):
    '''
        Models the atmosphere;
        holds the modelled Particle-Photon interactions and overall elemental composition
        has an optical length (thickness) a Photon would have to traverse in Monte-Carlo integration
        and density and temperature gradients that define variations in density and temperature
        at each optical depth point (see Cell)
    '''

    def __init__(self,
                 electron: 'Particle',
                 particles: set['Particle'],
                 transitions: set['Transition'],
                 elemental_composition: dict['Particle', float],
                 thickness: float,
                 density_gradient: Callable[[float|numpy.ndarray], float|numpy.ndarray],
                 temperature_gradient: Callable[[float|numpy.ndarray], float|numpy.ndarray],
                 core_density: float=None,
                 surface_density: float=None,
                 core_temperature: float=None,
                 surface_temperature: float=None):
        self.electron: 'Particle' = electron
        self.particles: set['Particle'] = particles
        self.transitions: set['Transition'] = transitions
        self.elemental_composition: dict['Particle', float] = elemental_composition
        self.thickness: float = thickness
        self.density_gradient: Callable[[float|numpy.ndarray], float|numpy.ndarray] = density_gradient
        self.temperature_gradient: Callable[[float|numpy.ndarray], float|numpy.ndarray] = temperature_gradient
        self.core_density: float = core_density
        self.surface_density: float = surface_density
        self.core_temperature: float = core_temperature
        self.surface_temperature: float = surface_temperature
    
    def cells(self, grid_size: float) -> numpy.ndarray:
        '''
            Integrates density and temperature gradients
            and models changes to elemental composition
            at each optical depth point along the thickness
            to instantiate and return Cells
        '''

        cell_positions = numpy.arange(0, self.thickness, grid_size) + (grid_size / 2)

        temperature_delta = self.temperature_gradient(cell_positions) * grid_size
        cumulative_temperature_delta = numpy.cumsum(temperature_delta)
        if self.core_temperature is not None:
            cell_temperatures = self.core_temperature + cumulative_temperature_delta
        else:
            cell_temperatures = self.surface_temperature - cumulative_temperature_delta

        density_delta = self.density_gradient(cell_positions) * grid_size
        cumulative_density_delta = numpy.cumsum(density_delta)
        if self.core_density is not None:
            cell_densities = self.core_density + cumulative_density_delta
        else:
            cell_densities = self.surface_density - cumulative_density_delta

        return numpy.array([DenseCell(self, cell_temperatures[i], cell_densities[i], self.elemental_composition)
                for i in range(cell_positions.size)])


class Gradient(Callable):
    '''
        A callable object that returns gradient value at a given position
        for modelling density and temperature gradients
    '''

    def value(self, position: float|numpy.ndarray) -> float|numpy.ndarray:
        return NotImplemented

    def __call__(self, position: float|numpy.ndarray) -> float|numpy.ndarray:
        return self.value(position)


class ConstantGradient(Gradient):
    '''
        A constant gradient function, g(x) = m

        Given initial and final values (f_i, f_l) over length (l)
        of the modelled function f(x) (= integral[g(x)]),

          m = (f_l - f_i) / l
    '''

    def __init__(self, initial_value: float, final_value: float, length: float):
        self.slope: float = (final_value - initial_value) / length

    def value(self, position: float|numpy.ndarray) -> float|numpy.ndarray:
        return numpy.full_like(position, self.slope) if isinstance(position, numpy.ndarray) else self.slope


class ZeroGradient(ConstantGradient):
    '''
        A constant gradient function, g(x) = 0
    '''

    def __init__(self):
        self.slope = 0.

class LinearGradient(Gradient):
    '''
        A linear gradient function, g(x) = ax + b

        Given the initial and final values (f_i, f_l) over length (l)
        of the modelled function f(x) (= integral[g(x)]),

          a = 2 * ((f_l - f_i) / l^2 - b / l)
    '''

    def __init__(self, initial_value: float, final_value: float, length: float, linear_coefficient=0.):
        self.linear_coefficient: float = linear_coefficient
        self.leading_coefficient: float = (2
                                           * (((final_value - initial_value) / length ** 2)
                                              - (self.linear_coefficient / length)))

    def value(self, position: float|numpy.ndarray) -> float|numpy.ndarray:
        return self.leading_coefficient * position + self.linear_coefficient

class ExponentialGradient(Gradient):
    '''
        An exponential gradient function, g(x) = a * exp(k * x)

        Given the initial and final values (f_i, f_l) over length (l)
        of the modelled function f(x) (= integral[g(x)]),

          k = (1 / l) * ln(f_l / f_i)
          a = k * f_i

        (ignoring the integration constant)
    '''

    def __init__(self, initial_value, final_value, length):
        self.exponent = numpy.log(final_value/initial_value) / length
        self.coefficient = initial_value * self.exponent

    def value(self, position: float|numpy.ndarray) -> float|numpy.ndarray:
        return self.coefficient * numpy.exp(self.exponent * position)


class Source(object):
    '''
        Models a source of photons
    '''

    def photons(self, sample_size: int) -> numpy.ndarray:
        '''
            Generates a sample of photons given by sample size
            Each photon may have a number of properties such as
            wavelength and direction of travel
        '''

        return NotImplemented


class BlackBodySource(Source):
    '''
        Models a Black Body radiation source
    '''

    PLANCK_RADIANCE_CONSTANT_TERM: float = 2 * scipy.constants.h * scipy.constants.c ** 2
    WIEN_DISPLACEMENT_CONSTANT_TERM: float = 2.897e-3

    def __init__(self, temperature: float, wavelength_bounding_box_scaling_factors: tuple[float, float]=(0.1, 8), excess: float=5.):
        self.temperature: float = temperature

        # Bounding box parameters for minimum and maximum wavelengths to generate
        # Default parameters span 99% of radiated energy
        self.wavelength_bounding_box_scaling_factors: tuple[float, float] = wavelength_bounding_box_scaling_factors
        # Multiplication factor for generating excess photons to compensate for rejection sampling losses
        self.excess: float = excess

        self.planck_radiance_exponent_constant_term: float = numpy.exp(scipy.constants.h * scipy.constants.c / (scipy.constants.k * temperature))

        # Use Wien's Displacement Law to find the bounding box parameter for maximum spectral radiance to generate
        self.wavelength_at_max_radiance: float = self.WIEN_DISPLACEMENT_CONSTANT_TERM / temperature
        self.max_spectral_radiance: float = self.spectral_radiance(self.wavelength_at_max_radiance)

    def spectral_radiance(self, wavelength: float) -> float:
        '''
            Spectral radiance given by Planck's law

              B(lambda, T) = (2 * h * c^2 / lambda^5) * (1 / (exp(h * c / (lambda * k * T)) - 1))
        '''

        return (self.PLANCK_RADIANCE_CONSTANT_TERM / (wavelength ** 5 * ((self.planck_radiance_exponent_constant_term ** (1 / wavelength)) - 1)))

    def photons(self, sample_size: int) -> numpy.ndarray:
        '''
            Generates photons described by wavelengths following Planck's law
            Returns an array of wavelengths

            Uses rejection sampling to generate photon wavelengths with an efficiency of 20%
            when using the default bounding box parameters

            Excess photons are generated (see excess parameter) to approximately
            match the required sample size
        '''
        sample_size_with_excess = int(sample_size * self.excess)

        # Generate wavelengths in bounding box limits
        wavelengths = numpy.random.uniform(*[limit * self.wavelength_at_max_radiance for limit in self.wavelength_bounding_box_scaling_factors],
                                           sample_size_with_excess)
        # Generate radiance values at each wavelength
        radiance = numpy.random.uniform(0, self.max_spectral_radiance, sample_size_with_excess)
        # Calculate maximum allowed radiance at each wavelength
        allowed_radiance = self.spectral_radiance(wavelengths)
        # Reject packets that exceed the maximum allowed radiance
        # as these cannot be generated by this Black Body object
        allowed_wavelengths = wavelengths[radiance < allowed_radiance]

        LOG.info(f'Sampled {sample_size_with_excess} photons; accepted {allowed_wavelengths.size}; required {sample_size}')

        return allowed_wavelengths[0:sample_size]


class SphericalVolumetricSource(Source):
    '''
        Extends a source to generate photons with
        a travel direction within a given angular span
    '''

    def __init__(self, source: 'Source', half_angular_span: float):
        self.source: 'Source' = source
        self.half_angular_span: float = half_angular_span
    
    def photons(self, sample_size: int) -> numpy.ndarray:
        '''
            Extends the photon wavelengths generated by the original source
            with direction vectors within an angular span

            Direction vectors point towards a point on surface of atmosphere
            where it meets horizontal axis with photon at the origin - not the center of source
        '''

        wavelengths = self.source.photons(sample_size)
        angular_direction = numpy.random.uniform(-self.half_angular_span, self.half_angular_span, wavelengths.shape)
        return numpy.vstack((wavelengths,
                             numpy.cos(angular_direction),
                             numpy.sin(angular_direction))).T


class Geometry(object):
    '''
        Models a geometry enclosing the photon source and the atmosphere

        Provides methods to advance photons through the geometry at each step
        of Monte-Carlo integration and map the photon positions to atmosphere Cells
    '''

    EMPTY_CELL = EmptyCell()

    def __init__(self, source: 'Source', source_span: float, atmosphere: 'Atmosphere', grid_size: float):
        self.source: 'Source' = source
        self.source_span: float = source_span
        self.atmosphere: 'Atmosphere' = atmosphere
        self.grid_size: float = grid_size

        self.atmosphere_cells = self.atmosphere.cells(self.grid_size)

    def positions(self, photons: numpy.ndarray, steps: int|numpy.ndarray=None, previous_positions: float|numpy.ndarray=None) -> float|numpy.ndarray:
        return NotImplemented
    
    def cells(self, positions: float|numpy.ndarray) -> Cell|numpy.ndarray:
        return NotImplemented
    

class PlanarGeometry(Geometry):
    '''
        Models 1D planar geometry (e.g. plane-parallel approximation of atmosphere)
    '''

    def positions(self, photons: numpy.ndarray, steps: int|numpy.ndarray=None, previous_positions: float|numpy.ndarray=None) -> float|numpy.ndarray:
        '''
            Moves photons by one grid per step and returns new positions

            Photon positions are modelled by a single co-ordinate
        '''

        return self.source_span + self.grid_size * steps if steps is not None else previous_positions + self.grid_size
    
    def cells(self, positions: float|numpy.ndarray) -> Cell|numpy.ndarray:
        '''
            Returns cells at photon positions
        '''
        
        return numpy.where((positions < self.source_span) & (positions > (self.source_span + self.atmosphere.thickness)),
                           self.EMPTY_CELL,
                           self.atmosphere_cells[numpy.asarray((positions - self.source_span)/self.grid_size, dtype=int)])


class SphericalGeometry(Geometry):
    '''
        Models 2D spherical geometry; all photons travel towards a single point on surface
    '''

    def __init__(self, source, source_span, atmosphere, grid_size):
        # Half angular span is the angle between horizontal axis
        # and tangent to source drawn from a point on surface of atmosphere
        # where it meets the horizontal axis
        # 
        # An observer at this point cannot see photons emitted at a larger angle
        # (ignoring the changes in direction due to scattering in 1D Monte-Carlo integration)
        self.source_half_angular_span = numpy.arcsin(source_span / (source_span + atmosphere.thickness))

        super().__init__(SphericalVolumetricSource(source, self.source_half_angular_span), source_span, atmosphere, grid_size)

    def positions(self, photons: numpy.ndarray, steps: int|numpy.ndarray=None, previous_positions: numpy.ndarray=None) -> numpy.ndarray:
        '''
            Moves a photon by grid_size distance per step

            Depending on angle at which the photon is emitted it may end up in the same grid it was in
        '''

        direction_vectors = photons[:, 1:]
        
        if steps is not None:
            # Direction vectors are towards a point on surface of atmosphere
            # with photon at origin - not the center of source
            # Use trigonometry to calculate initial position vectors
            # given the direction vectors
            initial_vectors = direction_vectors[:, 1::-1]
            upper_hemisphere_vectors = initial_vectors[:, 0] < 0
            initial_vectors[upper_hemisphere_vectors][0] *= -1
            initial_vectors[~upper_hemisphere_vectors][1] *= -1
            return self.source_span * initial_vectors + (self.grid_size * steps) * direction_vectors
        else:
            return previous_positions + self.grid_size * direction_vectors
    
    def cells(self, positions: numpy.ndarray) -> Cell|numpy.ndarray:
        '''
            Returns cells at photon positions

            Photons emitted at an angle take longer to escape atmosphere
            Empty cells are returned for photons that escaped atmosphere earlier than others
        '''

        radial_distances = numpy.linalg.norm(positions, axis=-1)
        return numpy.where((radial_distances < self.source_span) & (radial_distances > (self.source_span + self.atmosphere.thickness)),
                           self.EMPTY_CELL,
                           self.atmosphere_cells[numpy.asarray((radial_distances - self.source_span)/self.grid_size, dtype=int)])


if __name__ == '__main__':
    import argparse, os, json
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--file-prefix')
    parser.add_argument('-x', '--skip-simulation', action='store_true')
    args = parser.parse_args()

    from matplotlib import pyplot

    # Setup default figure size and use LaTeX for text formatting
    pyplot.rc('figure', figsize=[12, 8], dpi=144)
    pyplot.rc('text', usetex=True)

    # Load run configuration
    with open(f'{args.file_prefix}-config.json', 'r') as fyle:
        run_config = json.load(fyle)

    if not args.skip_simulation:
        # Load oscillator strength lookup table
        oscillator_strength = OscillatorStrength('oscillator_strengths.csv')

        # Setup a pure Hydrogen atmosphere and light-matter interactions
        electron = Particle('e', 0, 1, -1)
        particles = set()
        transitions = set([ThomsonScattering(electron, electron), FreeFreeAbsorption(electron, electron)])

        hydrogen_excitations = [Particle(f'H{i}', 1, i, 0) for i in range(1, 11)]
        hydrogen_ion = Particle('H+', 1, 1, 1)

        for particle in itertools.chain(hydrogen_excitations, [hydrogen_ion]):
            particles.add(particle)

        for i in range(len(hydrogen_excitations) - 1):
            for j in range(i + 1, len(hydrogen_excitations)):
                transitions.add(BoundBoundAbsorption(hydrogen_excitations[i],
                                                    hydrogen_excitations[j],
                                                    oscillator_strength.value(hydrogen_excitations[i].protons,
                                                                                hydrogen_excitations[i].charge,
                                                                                hydrogen_excitations[i].quantum_number,
                                                                                hydrogen_excitations[j].quantum_number)))
        for excitation in hydrogen_excitations:
            transitions.add(BoundFreeAbsorption(excitation, hydrogen_ion))
        
        # Setup temperature and density gradients and initialise atmosphere
        atmosphere_thickness = run_config["atmosphere"]["thickness"]

        if run_config["atmosphere"]["density_gradient"] == "zero":
            density_gradient = ZeroGradient()
        elif run_config["atmosphere"]["density_gradient"] == "linear":
            density_gradient = LinearGradient(run_config["atmosphere"]["core_density"],
                                              run_config["atmosphere"]["surface_density"],
                                              atmosphere_thickness)
        elif run_config["atmosphere"]["density_gradient"] == "exponential":
            density_gradient = ExponentialGradient(run_config["atmosphere"]["core_density"],
                                              run_config["atmosphere"]["surface_density"],
                                              atmosphere_thickness)
        else:
            raise ValueError(f'Unknown gradient type for density: {run_config["atmosphere"]["density_gradient"]}')

        if run_config["atmosphere"]["temperature_gradient"] == "zero":
            temperature_gradient = ZeroGradient()
        elif run_config["atmosphere"]["temperature_gradient"] == "linear":
            temperature_gradient = LinearGradient(run_config["atmosphere"]["core_temperature"],
                                              run_config["atmosphere"]["surface_temperature"],
                                              atmosphere_thickness)
        elif run_config["atmosphere"]["temperature_gradient"] == "exponential":
            temperature_gradient = ExponentialGradient(run_config["atmosphere"]["core_temperature"],
                                              run_config["atmosphere"]["surface_temperature"],
                                              atmosphere_thickness)
        else:
            raise ValueError(f'Unknown gradient type for temperature: {run_config["atmosphere"]["temperature_gradient"]}')
        
        grid_size = run_config["geometry"]["grid_size"]
        atmosphere = Atmosphere(electron, particles, transitions, {hydrogen_excitations[0]: 1.},
                                atmosphere_thickness, density_gradient, temperature_gradient,
                                core_density=run_config["atmosphere"]["core_density"],
                                core_temperature=run_config["atmosphere"]["core_temperature"])
        
        # Setup a Black Body radiation source with same temperature as at inner boundary of atmosphere
        source = BlackBodySource(run_config["source"]["temperature"])

        # Setup grid geometry
        if run_config["geometry"]["type"] == "spherical":
            # Spherical atmosphere
            geometry = SphericalGeometry(source, 0.66, atmosphere, grid_size)
            steps = numpy.arange(0, int(atmosphere_thickness/grid_size) + 1, 1)[:, numpy.newaxis, numpy.newaxis]
        else:
            # Plane parallel atmosphere
            geometry = PlanarGeometry(source, 0.66, atmosphere, grid_size)
            steps = numpy.arange(0, int(atmosphere_thickness/grid_size) + 1, 1)[:, numpy.newaxis]

        # Begin Monte-Carlo simulation

        # Generate photons
        input_photons = geometry.source.photons(run_config["source"]["photons"])

        # Handle previously chosen geometry
        is_directional_photon = len(input_photons.shape) > 1

        if is_directional_photon:
            input_wavelengths = input_photons[:, 0]
            input_directions = input_photons[:, 2] # Extract sine of angular direction
        else:
            input_wavelengths = input_photons
            input_directions = None
        
        # Save input photons
        numpy.save(f'{args.file_prefix}-input-wavelengths.npy', input_wavelengths)
        if is_directional_photon:
            numpy.save(f'{args.file_prefix}-input-directions.npy', input_directions)

        # For each photon calculate the position vector at each step
        # Result: (step, photon, co-ordinates)
        positions = geometry.positions(input_photons, steps=steps)

        # Map the position vectors to grid cells
        # Result: (step, photon, cell)
        cells = geometry.cells(positions)

        # For each wavelength calculate opacity it experiences at that step
        # Result: (step, photon, opacity)
        opacities = cells * input_wavelengths

        # Sample probabilities of absorption for each photon at each step
        # Result: (step, photon, absorption probability)
        # Collapse along step axis to check if photon is absorbed at any step
        # Result: (photon, is absorbed?)
        is_absorbed = numpy.any(numpy.random.random(opacities.shape) < opacities, axis=0)

        # Gather surviving photons
        output_photons = input_photons[~is_absorbed]

        # Extract surviving wavelengths (and directions at origin)
        if is_directional_photon:
            output_wavelengths = output_photons[:, 0]
            output_directions = output_photons[:, 2]
        else:
            output_wavelengths = output_photons
            output_directions = None

        # Save output photons
        numpy.save(f'{args.file_prefix}-output-wavelengths.npy', output_wavelengths)
        if is_directional_photon:
            numpy.save(f'{args.file_prefix}-output-directions.npy', output_directions)
    else:
        is_directional_photon = False
        input_directions = None
        output_directions = None
        input_wavelengths = numpy.load(f'{args.file_prefix}-input-wavelengths.npy')
        output_wavelengths = numpy.load(f'{args.file_prefix}-output-wavelengths.npy')
        if os.path.exists(f'{args.file_prefix}-input-directions.npy'):
            is_directional_photon = True
            input_directions = numpy.load(f'{args.file_prefix}-input-directions.npy')
            output_directions = numpy.load(f'{args.file_prefix}-output-directions.npy')

    # Analyse results and plot

    # Plot the emission spectrum
    pyplot.figure(1)
    pyplot.title(r'$Intensity~at~Wavelength$')
    pyplot.xlabel(r'$Wavelength~(m)$')
    pyplot.ylabel(r'$Intensity~(arbitrary~units)$')
    pyplot.hist(input_wavelengths, 1000, label=r'$Black~Body~Spectrum$')

    # Overlay the absorption spectrum
    pyplot.hist(output_wavelengths, 1000, label=r'$Atmospheric~Spectrum$')
    pyplot.legend()
    pyplot.show()

    if is_directional_photon:
        # Plot directional intensities from source
        pyplot.figure(2)
        pyplot.title(r'$Intensity~at~Angle$')
        pyplot.xlabel(r'$Angle~(rad)$')
        pyplot.ylabel(r'$Intensity~(arbitrary~units)$')
        pyplot.hist(numpy.arcsin(input_directions), 1000, label=r'$Source~Intensity$')

        # Overlay directional intensities past atmosphere
        pyplot.hist(numpy.arcsin(output_directions), 1000, label=r'$Atmospheric~Intensity$')
        pyplot.legend()
        pyplot.show()

    # Obtain and plot relative intensities
    def relative_intensity(input_signal: numpy.ndarray, output_signal: numpy.ndarray, bins: int=1000) -> tuple[numpy.ndarray, numpy.ndarray]:
        input_intensity, bin_edges = numpy.histogram(input_signal, bins=bins)
        output_intensity, output_bin_edges = numpy.histogram(output_signal, bins=bin_edges)
        intensity_ratio = output_intensity / input_intensity
        intensity_ratio[numpy.isnan(intensity_ratio)] = 1.
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, intensity_ratio
    
    # Plot relative intensity at each wavelength to demonstrate absorption in atmosphere
    wavelength_bins, relative_intensity_at_wavelength = relative_intensity(input_wavelengths, output_wavelengths)
    pyplot.figure(3)
    pyplot.title(r'$Relative~Intensity~at~Wavelength$')
    pyplot.xlabel(r'$Wavelength~(nm)$')
    pyplot.ylabel(r'$\frac{I_{atm}}{I_{bb}}~(dimensionless)$')
    pyplot.plot(wavelength_bins * 1e9, relative_intensity_at_wavelength)
    pyplot.show()

    if is_directional_photon:
        # Plot relative intensity in each direction to demonstrate Limb Darkening
        direction_bins, relative_intensity_in_direction = relative_intensity(numpy.arcsin(input_directions), numpy.arcsin(output_directions))
        pyplot.figure(4)
        pyplot.title(r'$Relative~Intensity~at~Angle$')
        pyplot.xlabel(r'$Angle~(rad)$')
        pyplot.ylabel(r'$\frac{I_{atm}}{I_{bb}}~(dimensionless)$')
        pyplot.plot(direction_bins, relative_intensity_in_direction)
        pyplot.show()
