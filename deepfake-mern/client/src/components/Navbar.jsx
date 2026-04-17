import { NavLink } from 'react-router-dom';

function Navbar() {
  return (
    <header className="sticky top-0 z-50 border-b border-slate-700/60 bg-[#0f172a]/85 backdrop-blur-md">
      <nav className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between px-4 md:px-6">
        <NavLink
          to="/"
          className="flex items-center gap-2 text-slate-100"
        >
          <span className="inline-flex h-9 w-9 items-center justify-center rounded-md bg-slate-800/80">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              className="h-5 w-5 text-blue-400"
              aria-hidden="true"
            >
              <path d="M12 3 4.5 6v6.6c0 4.2 3.06 8.12 7.5 9.4 4.44-1.28 7.5-5.2 7.5-9.4V6L12 3Z" />
              <path d="m9.4 12.4 1.8 1.8 3.6-3.6" />
            </svg>
          </span>
          <span className="text-base font-semibold tracking-wide">DeepFake Detector</span>
        </NavLink>
      </nav>
    </header>
  );
}

export default Navbar;
