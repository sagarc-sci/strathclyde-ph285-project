import numpy, scipy, pandas
import itertools, logging, sys
from functools import cache


LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler(sys.stderr))
LOG.setLevel(logging.INFO)


class OscillatorStrengths(object):
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
            # Quantum mechanical correction factor
            # Not implemented
            gaunt_factor = 1.
            # Menzel-Pekeris Approximation
            # Only true for Hydrogenic species
            return (((32 / (3 * numpy.pi * numpy.sqrt(3))) / (quantum_number ** 5 * excited_quantum_number ** 3))
                    * ((1 / quantum_number ** 2) - (1 / excited_quantum_number ** 2)) ** (-3)
                    * gaunt_factor)


class Atmosphere(object):
    def __init__(self, electron: 'Particle', particles: set['Particle'], transitions: set['Transition']):
        self.electron: 'Particle' = electron
        self.particles: set['Particle'] = particles
        self.transitions: set['Transition'] = transitions


class Cell(object):
    def __init__(self, atmosphere: 'Atmosphere', temperature: float, number_density: float, elemental_composition: dict['Particle', float]):
        self.atmosphere: 'Atmosphere' = atmosphere
        self.temperature: float = temperature
        self.number_density: float = number_density
        self.elemental_composition: dict['Particle', float] = elemental_composition
        self.particle_number_densities: dict['Particle', float] = self.solve_particle_number_densities()

    def solve_particle_number_densities(self) -> dict['Particle', float]:
        particle_number_densities: dict['Particle', float] = dict()

        ion_number_densities, electron_number_density = self.solve_ion_number_densities()

        particle_number_densities[self.atmosphere.electron] = electron_number_density

        for ion, ion_number_density in ion_number_densities.items():
            particle_number_densities.update(self.solve_excitation_number_densities(ion, ion_number_density))
        
        return particle_number_densities

    def solve_ion_number_densities(self, max_electron_density_error: float=0.01, max_iterations: int=100) -> tuple[dict['Particle', float], float]:
        ions_of_element: dict['Particle', list['Particle']] = dict()
        for element in self.elemental_composition:
            ions = set([element])
            for next_function in [Particle.previous_ion, Particle.next_ion]:
                ion = element
                while (ion := next_function(ion)):
                    ion = ion.ground_state()
                    ions.add(ion)
            ions_of_element[element] = sorted(ions, key=lambda ion: ion.charge)

        partition_function_values: dict['Particle', float] = dict()
        for ions in ions_of_element.values():
            for ion in ions:
                partition_function_values[ion] = ion.partition_function(self.temperature)
                excitation = ion
                while(excitation := excitation.next_excitation()):
                    partition_function_values[ion] += excitation.partition_function(self.temperature)

        ion_number_densities: dict['Particle', float] = \
            dict((ion, self.elemental_composition[element] * self.number_density / len(ions))
                 for element, ions in ions_of_element.items()
                 for ion in ions)
        electron_number_density = 0.
        electron_density_converged = False
        constant_term = 2 * ((2 * numpy.pi * scipy.constants.m_e * scipy.constants.k) ** 1.5) / (scipy.constants.h ** 3)
        for iteration in range(max_iterations):
            updated_electron_number_density = sum(ion.charge * ion_number_density
                                                  for ion, ion_number_density in ion_number_densities.items())
            if numpy.isclose(electron_number_density, updated_electron_number_density, rtol=max_electron_density_error):
                electron_number_density = updated_electron_number_density
                electron_density_converged = True
                break
            electron_number_density = updated_electron_number_density
            for element, ions in ions_of_element.items():
                for i in range(1, len(ions)):
                    ion_number_densities[ions[i]] = ((constant_term
                     * (partition_function_values[ions[i]] / partition_function_values[ions[i - 1]])
                     * (self.temperature ** 1.5)
                     * numpy.exp(- ions[i - 1].ionisation_energy() / (scipy.constants.k * self.temperature)))
                     * (ion_number_densities[ions[i - 1]] / electron_number_density))
                scaling_factor = (self.elemental_composition[element] * self.number_density
                                  / sum(ion_number_densities[ion] for ion in ions))
                for ion in ions:
                    ion_number_densities[ion] *= scaling_factor
        if not electron_density_converged:
            LOG.warning('Electron number density did not converge')

        return ion_number_densities, electron_number_density
    
    def solve_excitation_number_densities(self, particle: 'Particle', particle_number_density: float) -> dict['Particle', float]:
        partition_function_values: dict['Particle', float] = dict()
        partition_function_values[particle] = particle.partition_function(self.temperature)
        excitation = particle
        while (excitation := excitation.next_excitation()):
            partition_function_values[excitation] = excitation.partition_function(self.temperature)
        scaling_factor = particle_number_density / sum(partition_function_values.values())

        excitation_number_densities: dict['Particle', float] = \
            dict((excitation, partition_function_value * scaling_factor)
                 for excitation, partition_function_value in partition_function_values.items())
        
        return excitation_number_densities

    def opacity(self, wavelength: float):
        return sum(transition.opacity(wavelength, self) for transition in self.atmosphere.transitions)


class Transition(object):
    def __init__(self, source: 'Particle', destination: 'Particle'):
        self.source: 'Particle' = source
        self.destination: 'Particle' = destination

        self.source.add_outgoing_transition(self)
        self.destination.add_incoming_transition(self)
    
    def symbol(self) -> str:
        return NotImplemented
    
    def __repr__(self) -> str:
        return f'{self.source} -- {self.symbol()} --> {self.destination}'
    
    def cross_section(self, wavelength: float, cell: 'Cell') -> float:
        return NotImplemented
    
    def opacity(self, wavelength: float, cell: 'Cell') -> float:
        return cell.particle_number_densities[self.source] * self.cross_section(wavelength, cell)


class ThomsonScattering(Transition):
    def symbol(self):
        return 'TH'
    
    def cross_section(self, wavelength: float, cell: 'Cell') -> float:
        return 6.65e-29 # m2


class FreeFreeAbsorption(Transition):
    CONSTANT_TERM = ((numpy.sqrt(32 * numpy.pi) * scipy.constants.e ** 6)
                     / ((3 * numpy.sqrt(3) * scipy.constants.h * scipy.constants.c ** 4)
                        * numpy.sqrt(scipy.constants.k * scipy.constants.m_e ** 3)))

    def symbol(self) -> str:
        return 'FF'

    def gaunt_factor(self, wavelength: float, cell: 'Cell') -> float:
        # Quantum mechanical correction factor
        # Not implemented
        return 1.

    def cross_section(self, wavelength: float, cell: 'Cell') -> float:
        # Only true for Hydrogenic species
        return (self.CONSTANT_TERM
                * (wavelength ** 3 / numpy.sqrt(cell.temperature))
                * sum(particle_number_density * particle.protons ** 2
                      for particle, particle_number_density in cell.particle_number_densities.items()
                      if particle.charge != 0 and particle is not cell.atmosphere.electron)
                * self.gaunt_factor(wavelength, cell))


class BoundFreeAbsorption(Transition):
    def __init__(self, source: 'Particle', destination: 'Particle'):
        super().__init__(source, destination)
        self.constant_term = (((64 * (numpy.pi ** 4) * (scipy.constants.e ** 10) * scipy.constants.m_e)
                              / (3 * numpy.sqrt(3) * (scipy.constants.c ** 4) * (scipy.constants.h ** 6)
                                 * ((4 * numpy.pi * scipy.constants.epsilon_0) ** 5)))
                                 * (self.source.protons ** 4 / self.source.quantum_number ** 5))

    def symbol(self) -> str:
        return 'BF'

    def gaunt_factor(self, wavelength: float, cell: 'Cell') -> float:
        # Quantum mechanical correction factor
        # Not implemented
        return 1.

    def cross_section(self, wavelength: float, cell: 'Cell') -> float:
        # Only true for Hydrogenic species
        return numpy.where(wavelength <= self.source.ionisation_wavelength(),
                           self.constant_term * (wavelength ** 3) * self.gaunt_factor(wavelength, cell),
                           0.)


class BoundBoundAbsorption(Transition):
    def __init__(self, source: 'Particle', destination: 'Particle', oscillator_strength: float):
        super().__init__(source, destination)
        self.line_wavelength = 1 / ((1 / self.source.ionisation_wavelength()) - (1 / self.destination.ionisation_wavelength()))
        self.line_frequency = scipy.constants.c / self.line_wavelength
        self.cross_section_constant_term = ((scipy.constants.e ** 2 / (4
                                                                       * scipy.constants.epsilon_0
                                                                       * scipy.constants.m_e
                                                                       * scipy.constants.c))
                                            * oscillator_strength)
        self.doppler_width_constant_term = (numpy.sqrt(2 * scipy.constants.k / (scipy.constants.m_p * self.source.protons))
                                            / self.line_wavelength)
                                 
    def symbol(self) -> str:
        return 'BB'
    
    def doppler_profile(self, wavelength: float, cell: 'Cell') -> float:
        doppler_width = self.doppler_width_constant_term * numpy.sqrt(cell.temperature)
        frequency = scipy.constants.c / wavelength
        return numpy.exp(- ((frequency - self.line_frequency) / doppler_width) ** 2) / (doppler_width * numpy.sqrt(numpy.pi))
    
    def line_profile(self, wavelength: float, cell: 'Cell') -> float:
        return self.doppler_profile(wavelength, cell)

    def gaunt_factor(self, wavelength: float, cell: 'Cell') -> float:
        # Quantum mechanical correction factor
        # Not implemented
        return 1.

    def cross_section(self, wavelength: float, cell: 'Cell'):
        return (self.cross_section_constant_term
                * (1 - ((self.source.degeneracy() * cell.particle_number_densities[self.destination])
                        / (self.destination.degeneracy() * cell.particle_number_densities[self.source])))
                * self.line_profile(wavelength, cell)
                * self.gaunt_factor(wavelength, cell))


class Particle(object):
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
        self.outgoing_transitions.add(transition)
        return self

    def add_incoming_transition(self, transition: 'Transition') -> 'Particle':
        self.incoming_transitions.add(transition)
        return self
    
    @cache
    def ground_state(self) -> 'Particle':
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
        return max((transition.source for transition in self.incoming_transitions if isinstance(transition, BoundBoundAbsorption)),
                   key=lambda particle: particle.quantum_number, default=None)

    @cache
    def next_excitation(self) -> 'Particle':
        return min((transition.destination for transition in self.outgoing_transitions if isinstance(transition, BoundBoundAbsorption)),
                   key=lambda particle: particle.quantum_number, default=None)
    
    @cache
    def elemental_state(self) -> 'Particle':
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
        for transition in self.incoming_transitions:
            if isinstance(transition, BoundFreeAbsorption):
                return transition.source
        return None
    
    @cache
    def next_ion(self) -> 'Particle':
        for transition in self.outgoing_transitions:
            if isinstance(transition, BoundFreeAbsorption):
                return transition.destination
        return None
    
    @cache
    def degeneracy(self) -> float:
        # Only true for Hydrogenic species
        return 2 * self.quantum_number ** 2
    
    @cache
    def ionisation_energy(self) -> float:
        # Only true for Hydrogenic species
        return scipy.constants.physical_constants['Rydberg constant times hc in J'][0] * (self.protons / self.quantum_number) ** 2

    @cache
    def ionisation_wavelength(self) -> float:
        return (scipy.constants.h * scipy.constants.c) / self.ionisation_energy()
    
    @cache
    def partition_function(self, temperature: float) -> float:
        if self.protons - self.charge == 0:
            return 1.
        return self.degeneracy() * numpy.exp((self.ionisation_energy() - self.ground_state().ionisation_energy())
                                             / (scipy.constants.k * temperature))


if __name__ == '__main__':
    oscillator_strengths = OscillatorStrengths('oscillator_strengths.csv')

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
                                                 oscillator_strengths.value(hydrogen_excitations[i].protons,
                                                                            hydrogen_excitations[i].charge,
                                                                            hydrogen_excitations[i].quantum_number,
                                                                            hydrogen_excitations[j].quantum_number)))
    for excitation in hydrogen_excitations:
        transitions.add(BoundFreeAbsorption(excitation, hydrogen_ion))
    
    atmosphere = Atmosphere(electron, particles, transitions)
    cell = Cell(atmosphere, 20000, 1e20, {hydrogen_excitations[0]: 1.})

    print(cell.particle_number_densities)
    print(cell.opacity(121.5e-9))