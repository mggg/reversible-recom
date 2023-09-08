import matplotlib.pyplot as plt
# each line is 100,000 accepted steps
steps_per_line = 100_000
state_space_size = 158_753_814
rows = [
    (int(line.split()[0]), int(line.split()[1]))
    for line in open('outputs/7x7_7_seed_2_94915664_cover_10b.log')
]
plt.plot(
    [100 * idx * steps_per_line / state_space_size
     for idx in range(len(rows))],
    [100 * y / state_space_size for _, y in rows]
)
# plt.ylabel('% state space seen')
# plt.xlabel('Accepted steps (as % of state space size)')
plt.savefig('7x7_7_seed_2_94915664_cover_10b_rel_no_labels.pdf', bbox_inches='tight')
