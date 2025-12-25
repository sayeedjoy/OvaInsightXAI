import Image from "next/image";

const supervisor = {
  name: "Khandaker Mohammad Mohi Uddin",
  title: "Supervisor",
  imageUrl: "/team/jilani.webp",
};

const teamMembers = [
  {
    name: "Zubayer Ahmad Shibly",
    title: "Lead Researcher & ML Model Developer",
    imageUrl: "/team/shibly.webp",
  },
  {
    name: "Sheikh Arman Karim Aditto",
    title: "Manuscript Lead Co-Author",
    imageUrl: "/team/aditto.webp",
  },
  {
    name: "Jahidul Islam Asif",
    title: "Manuscript Editor",
    imageUrl: "/team/asif.webp",
  },
  {
    name: "Sayeed Joy",
    title: "Full-Stack & ML Ops",
    imageUrl: "/team/joy.webp",
  },
];

const Team = () => {
  return (
    <div className="flex flex-col items-center justify-center py-14 px-4 sm:px-6 lg:px-8">
      <div className="text-center max-w-xl mx-auto">
        <h2 className="mt-4 text-4xl sm:text-5xl font-semibold tracking-tighter">
          Meet Our Team
        </h2>
      </div>

      {/* Supervisor Section */}
      <div className="mt-20 flex justify-center">
        <div className="text-center">
          <Image
            src={supervisor.imageUrl}
            alt={supervisor.name}
            className="h-20 w-20 rounded-full object-cover mx-auto bg-secondary"
            width={120}
            height={120}
          />
          <h3 className="mt-4 text-lg font-semibold">{supervisor.name}</h3>
          <p className="text-muted-foreground">{supervisor.title}</p>
        </div>
      </div>

      {/* Team Members Grid */}
      <div className="mt-12 w-full grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-12 max-w-(--breakpoint-lg) mx-auto">
        {teamMembers.map((member) => (
          <div key={member.name} className="text-center">
            <Image
              src={member.imageUrl}
              alt={member.name}
              className="h-20 w-20 rounded-full object-cover mx-auto bg-secondary"
              width={120}
              height={120}
            />
            <h3 className="mt-4 text-lg font-semibold">{member.name}</h3>
            <p className="text-muted-foreground">{member.title}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Team;
